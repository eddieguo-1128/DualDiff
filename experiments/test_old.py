from models import *
from dataset import *
import pickle
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
    top_k_accuracy_score)


# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='Path to the trained model (.pth file)')
parser.add_argument('--data_loader_path', type=str, help='Path to the pickled DataLoader file')
args = parser.parse_args()

# Define model parameters (must match training setup)
n_T = 1000
ddpm_dim = 128
encoder_dim = 256
fc_dim = 512
# Define model
num_classes = 26
channels = 64  # Change this if your data has different channel count

# Instantiate the model
encoder = EEGNet(nb_classes=num_classes, Chans=channels, Samples=250, dropoutRate=0.2, kernLength=64, F1=16, D=2, F2=32, dropoutType='Dropout').to(device)
decoder = Decoder(in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim).to(device)
fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
diffe = DiffE(encoder, decoder, fc).to(device)

# load the pre-trained model from the file
diffe.load_state_dict(torch.load(args.model_path))

# load the data loader from the file
with open(args.data_loader_path, 'rb') as f:
    data_loader = pickle.load(f)

test1_loader = data_loader["test1"]
test2_loader = data_loader["test2"]



def get_subjectwise_z_stats_from_loader(loader, encoder, device, num_sessions=6):
    """
    For each subject in the given loader, compute the mean and std of their z
    using samples from the first 4 sessions only.

    Returns:
        A dictionary: {sid: (mean, std)}, where each value has shape [1, D]
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

    # Compute mean and std using only the first 104 samples (4 sessions × 26 samples/session)
    samples_per_session = 26
    z_stats = {}
    for sid in z_by_sid:
        z_all = torch.cat(z_by_sid[sid], dim=0)  # total: 156
        z_4sessions = z_all[:samples_per_session * 4]  # first 4 session (104 in total)
        mean = z_4sessions.mean(dim=0, keepdim=True)
        std = z_4sessions.std(dim=0, keepdim=True) + 1e-6
        z_stats[sid] = (mean, std)
    return z_stats

def evaluate_on_loader(test_loader, name="test1", z_stats=None):
    diffe.eval()
    labels = np.arange(0, 26)
    Y, Y_hat = [], []
    with torch.no_grad():
        for x, y,sid in test_loader:
            x, y = x.to(device), y.type(torch.LongTensor).to(device)
            encoder_out = diffe.encoder(x)
            z = encoder_out[1]
            z = torch.stack([
                (z[i] - z_stats[int(sid[i].item())][0].squeeze(0)) /
                z_stats[int(sid[i].item())][1].squeeze(0)
                for i in range(z.size(0))
            ])

            y_hat = diffe.fc(encoder_out[1])
            y_hat = F.softmax(y_hat, dim=1)

            Y.append(y.detach().cpu())
            Y_hat.append(y_hat.detach().cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = torch.cat(Y_hat, dim=0).numpy()

    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
    print(f" {name} Accuracy: {accuracy:.2%}")
    return accuracy

def evaluate_on_loader_test2(test_loader, name="test2", num_subjects=2, num_sessions=6):
    diffe.eval()
    labels = np.arange(0, 26)
    Y, Y_hat = [], []
    with torch.no_grad():
        all_x, all_y = [], []
        for x, y, _ in test_loader:
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0).to(device)
        all_y = torch.cat(all_y, dim=0).to(device)

        samples_per_subject = num_sessions * 26  # Each subject has 6 sessions × 26 samples = 156
        for i in range(num_subjects):
            start = i * samples_per_subject
            end = (i + 1) * samples_per_subject
            x_sub = all_x[start:end]
            y_sub = all_y[start:end]

            encoder_out = diffe.encoder(x_sub)
            z = encoder_out[1]  # shape: [156, 256]

            # Use the first 104 samples (sessions 0–3) to compute mean and std
            z_mean = z[:104].mean(dim=0, keepdim=True)
            z_std = z[:104].std(dim=0, keepdim=True) + 1e-6

            print(f"[Check] Subject {i+1}: z_mean shape {z_mean.shape}, z_std shape {z_std.shape}")

            z = (z - z_mean) / z_std  # normalize all 156 samples

            y_hat = diffe.fc(z)
            y_hat = F.softmax(y_hat, dim=1)

            Y.append(y_sub.detach().cpu())
            Y_hat.append(y_hat.detach().cpu())

    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = torch.cat(Y_hat, dim=0).numpy()
    accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
    print(f" {name} Accuracy: {accuracy:.2%}")
    return accuracy


z_stats = get_subjectwise_z_stats_from_loader(test2_loader, diffe.encoder, device)
acc1 = evaluate_on_loader(test1_loader, name="Test1 (Seen Subject)", z_stats=z_stats)
acc2 = evaluate_on_loader_test2(test2_loader, name="Test2 (Unseen Subject)")
