from models import *
from utils import *

import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [B, D] — z output from the encoder
        labels:   [B]    — integer type labels
        """
        device = features.device
        features = F.normalize(features, dim=1)              # Feature normalization
        batch_size = features.shape[0]

        # Construct positive sample mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # [B, B], 1 if same class

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # [B, B]

        # Exclude diagonal (self with self)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask
        sim = sim - 1e9 * (1 - logits_mask)  # Mask the diagonal with large negative value

        # Compute log-softmax
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)

        # Compute mean log-probability of positive samples for each instance
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # Mean negative log-likelihood
        loss = -mean_log_prob_pos.mean()
        return loss

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=1)

def get_subjectwise_z_stats(loader, encoder, device, num_sessions=6):
    """
    Compute z_mean and z_std for each subject using only sessions 0–3.

    Returns:
        A dictionary: {subject_id: (mean, std)} where each value has shape [1, D]

    Also prints whether each subject used exactly 104 samples
    (i.e., 4 sessions * 26 samples/session).
    """
    encoder.eval()
    z_by_sid = {}

    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            _, z = encoder(x)
            for i in range(z.size(0)):
                s = int(sid[i].item())
                if s not in z_by_sid:
                    z_by_sid[s] = []
                z_by_sid[s].append(z[i].unsqueeze(0))

    z_stats = {}
    for sid in sorted(z_by_sid):
        z_cat = torch.cat(z_by_sid[sid], dim=0)  # shape: [N, 256]
        print(f"[Check] Subject {sid}: z samples = {z_cat.shape[0]} (expected: {26 * 4})")
        assert z_cat.shape[0] >= 104, f"Subject {sid} z count < 104 — check loader input!"
        z_4session = z_cat[:26 * 4]  # only take the first 104 
        z_mean = z_4session.mean(dim=0, keepdim=True)
        z_std = z_4session.std(dim=0, keepdim=True) + 1e-6
        z_stats[sid] = (z_mean, z_std)
    return z_stats
    
# Evaluate function
def evaluate(encoder, fc, generator, device):
    labels = np.arange(0, 26)
    Y = []
    Y_hat = []
    for x, y,sid in generator:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        encoder_out = encoder(x)
        y_hat = fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)

        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

    # List of tensors to tensor to numpy
    Y = torch.cat(Y, dim=0).numpy()  # (N, )
    Y_hat = torch.cat(Y_hat, dim=0).numpy()  # (N, 13): has to sum to 1 for each row

    # Accuracy and Confusion Matrix
    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
    f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)

    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "auc": auc,
    }
    # df_cm = pd.DataFrame(confusion_matrix(Y, Y_hat.argmax(axis=1)))
    return metrics

def train(args):
    subject = args.subject
    device = args.device
    device = torch.device(device)
    batch_size = 32
    batch_size2 = 260
    seed = 44
    random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed: ", seed)

    # EEG data path
    root_dir = "Path-to-the-data"
    # Write performance metrics to file
    # output_dir = "performance-metric-path"
    # output_file = f"{output_dir}/{subject}.txt"

    # Load data
    loaders = load_split_dataset(root_dir=root_dir, num_seen=33, session=1)
    # Dataloader
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    #val2_loader = loaders["val2"]
    test1_loader = loaders["test1"]
    test2_loader = loaders["test2"]
    train_iter = iter(train_loader)

    # Define model
    num_classes = 26
    channels = X.shape[1]

    n_T = 1000
    ddpm_dim = 128
    encoder_dim = 256
    fc_dim = 512

    ddpm_model = ConditionalUNet(in_channels=channels, n_feat=ddpm_dim).to(device)
    ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(
        device
    )
    encoder = EEGNet(nb_classes=num_classes, Chans=channels, Samples=250, dropoutRate=0.2, kernLength=64, F1=16, D=2, F2=32, dropoutType='Dropout').to(device)
    decoder = Decoder(
        in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim
    ).to(device)
    fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    diffe = DiffE(encoder, decoder, fc).to(device)

    print(" Model initialized with:")
    print(" - Input channels       :", channels)
    print(" - Timepoints           :", timepoints)
    print(" - ddpm total params    :", sum(p.numel() for p in ddpm.parameters()))
    print(" - encoder total params :", sum(p.numel() for p in encoder.parameters()))
    print(" - decoder total params :", sum(p.numel() for p in decoder.parameters()))
    print(" - classifier params    :", sum(p.numel() for p in fc.parameters()))
    print(" - total DiffE params   :", sum(p.numel() for p in diffe.parameters()))


    # Criterion
    criterion = nn.L1Loss()
    criterion_class = nn.CrossEntropyLoss()

    # Define optimizer
    base_lr, lr = 9e-5, 1.5e-3
    optim1 = optim.RMSprop(ddpm.parameters(), lr=base_lr)
    optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)

    # EMAs
    fc_ema = EMA(diffe.fc, beta=0.95, update_after_step=100, update_every=10,)

    step_size = 150
    scheduler1 = optim.lr_scheduler.CyclicLR(
        optimizer=optim1,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    scheduler2 = optim.lr_scheduler.CyclicLR(
        optimizer=optim2,
        base_lr=base_lr,
        max_lr=lr,
        step_size_up=step_size,
        mode="exp_range",
        cycle_momentum=False,
        gamma=0.9998,
    )
    # Train & Evaluate
    num_epochs = 500
    test_period = 1
    start_test = test_period

    z_stats = get_subjectwise_z_stats(train_loader, diffe.encoder, device)
    supcon_loss = SupConLoss(temperature=0.07)
    proj_head = ProjectionHead(input_dim=256, proj_dim=128).to(device)

    best_acc = 0
    best_f1 = 0
    best_recall = 0
    best_precision = 0
    best_auc = 0

    with tqdm(
        total=num_epochs, desc=f"Method ALL - Processing subject {subject}"
    ) as pbar:
        for epoch in range(num_epochs):
            ddpm.train()
            diffe.train()

            epoch_loss = 0
            num_batches = 0
            epoch_acc = 0
            total_samples = 0

            alpha = 1
            beta = min(1.0, epoch / 50) * 0.2
            gamma = min(1.0, epoch / 100) * 0.05

            ############################## Train ###########################################
            for x, y,sid in train_loader:
                x, y = x.to(device), y.type(torch.LongTensor).to(device)
                y_cat = F.one_hot(y, num_classes=26).type(torch.FloatTensor).to(device)
                # Train DDPM
                optim1.zero_grad()
                x_hat, down, up, noise, t = ddpm(x)

                # Align the temporal dimension of x_hat and x
                if x_hat.shape[-1] != x.shape[-1]:
                    target_len = min(x_hat.shape[-1], x.shape[-1])
                    x_hat = F.interpolate(x_hat, size=target_len)
                    x = F.interpolate(x, size=target_len)

                loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                loss_ddpm.mean().backward()
                optim1.step()
                ddpm_out = x_hat, down, up, t

                # Train Diff-E
                optim2.zero_grad()
                decoder_out, fc_out,z = diffe(x, ddpm_out)#return Z
                z = torch.stack([(z[i] - z_stats[int(sid[i].item())][0].squeeze(0)) / z_stats[int(sid[i].item())][1].squeeze(0) for i in range(z.size(0))])

                loss_gap = criterion(decoder_out, loss_ddpm.detach())
                loss_decoder = F.l1_loss(decoder_out, x_hat.detach())
                #loss_c = criterion_class(fc_out, y_cat)
                loss_c = criterion_class(fc_out, y)
                z_proj = proj_head(z)
                loss_supcon = supcon_loss(z_proj, y)
                #loss = loss_gap + alpha * loss_c
                loss = alpha * loss_c + beta * loss_supcon + gamma * loss_decoder
                loss.backward()
                optim2.step()

                # Optimizer scheduler step
                scheduler1.step()
                scheduler2.step()

                # EMA update
                fc_ema.update()

                epoch_loss += loss.item()
                num_batches += 1

                pred_labels = torch.argmax(fc_out, dim=1)
                correct = (pred_labels == y).sum().item()
                epoch_acc += correct
                total_samples += y.size(0)

            ############################## Test ###########################################
            with torch.no_grad():
                if epoch > start_test:
                    test_period = 1
                if epoch % test_period == 0:
                    ddpm.eval()
                    diffe.eval()

                    #metrics_val = evaluate(diffe.encoder, fc_ema, val_loader, device)
                    metrics_val = evaluate(diffe.encoder, diffe.fc, val_loader, device)

                    val_acc = metrics_val["accuracy"]
                    history["val_acc"].append(val_acc)
                    f1 = metrics_val["f1"]
                    recall = metrics_val["recall"]
                    precision = metrics_val["precision"]
                    auc = metrics_val["auc"]

                    val_loss = 0
                    with torch.no_grad():
                        for x, y,sid in val_loader:
                            x, y = x.to(device), y.type(torch.LongTensor).to(device)
                            y_cat = F.one_hot(y, num_classes=26).float().to(device)

                            x_hat, down, up, noise, t = ddpm(x)
                            ddpm_out = x_hat, down, up, t

                            if x_hat.shape[-1] != x.shape[-1]:
                                target_len = min(x_hat.shape[-1], x.shape[-1])
                                x_hat = F.interpolate(x_hat, size=target_len)
                                x = F.interpolate(x, size=target_len)

                            loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
                            decoder_out, fc_out,z = diffe(x, ddpm_out)
                            z = torch.stack([(z[i] - z_stats[int(sid[i].item())][0].squeeze(0)) / z_stats[int(sid[i].item())][1].squeeze(0) for i in range(z.size(0))])

                            loss_gap = criterion(decoder_out, loss_ddpm)
                            loss_decoder = F.l1_loss(decoder_out, x_hat.detach())
                            #loss_c = criterion_class(fc_out, y_cat)
                            loss_c = criterion_class(fc_out, y)
                            z_proj = proj_head(z)
                            loss_supcon = supcon_loss(z_proj, y)

                            #val_loss += (loss_gap + alpha * loss_c).item()
                            val_loss += (alpha * loss_c + beta * loss_supcon+ gamma * loss_decoder).item()
                    history["val_loss"].append(val_loss / len(val_loader))

                    best_acc_bool = val_acc > best_acc
                    best_f1_bool = f1 > best_f1
                    best_recall_bool = recall > best_recall
                    best_precision_bool = precision > best_precision
                    best_auc_bool = auc > best_auc

                    if best_acc_bool:
                        best_acc = val_acc
                        #torch.save(diffe.state_dict(), f'/content/drive/MyDrive/project/model/ssvep/diffe_{subject}.pth')
                        torch.save(diffe.state_dict(), '/content/drive/MyDrive/project/model/ssvep/diffe_loss_z-norm2_attention.pth')
                    if best_f1_bool:
                        best_f1 = f1
                    if best_recall_bool:
                        best_recall = recall
                    if best_precision_bool:
                        best_precision = precision
                    if best_auc_bool:
                        best_auc = auc

                    # print("Subject: {0}".format(subject))
                    # # print("ddpm test loss: {0:.4f}".format(t_test_loss_ddpm/len(test_generator)))
                    # # print("encoder test loss: {0:.4f}".format(t_test_loss_ed/len(test_generator)))
                    # print("accuracy:  {0:.2f}%".format(acc*100), "best: {0:.2f}%".format(best_acc*100))
                    # print("f1-score:  {0:.2f}%".format(f1*100), "best: {0:.2f}%".format(best_f1*100))
                    # print("recall:    {0:.2f}%".format(recall*100), "best: {0:.2f}%".format(best_recall*100))
                    # print("precision: {0:.2f}%".format(precision*100), "best: {0:.2f}%".format(best_precision*100))
                    # print("auc:       {0:.2f}%".format(auc*100), "best: {0:.2f}%".format(best_auc*100))
                    # writer.add_scalar(f"EEGNet/Accuracy/subject_{subject}", acc*100, epoch)
                    # writer.add_scalar(f"EEGNet/F1-score/subject_{subject}", f1*100, epoch)
                    # writer.add_scalar(f"EEGNet/Recall/subject_{subject}", recall*100, epoch)
                    # writer.add_scalar(f"EEGNet/Precision/subject_{subject}", precision*100, epoch)
                    # writer.add_scalar(f"EEGNet/AUC/subject_{subject}", auc*100, epoch)

                    # if best_acc_bool or best_f1_bool or best_recall_bool or best_precision_bool or best_auc_bool:
                    #     performance = {'subject': subject,
                    #                 'epoch': epoch,
                    #                 'accuracy': best_acc*100,
                    #                 'f1_score': best_f1*100,
                    #                 'recall': best_recall*100,
                    #                 'precision': best_precision*100,
                    #                 'auc': best_auc*100
                    #                 }
                    #     with open(output_file, 'a') as f:
                    #         f.write(f"{performance['subject']}, {performance['epoch']}, {performance['accuracy']}, {performance['f1_score']}, {performance['recall']}, {performance['precision']}, {performance['auc']}\n")
                    description = f"Val Accuracy: {val_acc*100:.2f}% | Best: {best_acc*100:.2f}%"
                    pbar.set_description(f"Method ALL – Processing subject {subject} – {description}"
                    )
            print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {history['train_loss'][-1]:.4f} | Val Acc: {val_acc*100:.2f}%")
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model")
    # Define command-line arguments
    parser.add_argument(
        "--num_subjects", type=int, default=22, help="number of subjects to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use (default: cuda:0)"
    )

    # Parse command-line arguments
    args = parser.parse_args()
    for i in range(2, args.num_subjects + 1):
        subject = i
        args.subject = subject
        train(args)
