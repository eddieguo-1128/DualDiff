from sklearn.metrics import accuracy_score
from config import *
from dataset import *
from loss import *
from models import *
from utils import *
from viz import *
z_local_norm_mode = os.environ.get("Z_LOCAL_NORM_MODE", "option1")

def evaluate(encoder, fc, generator, device, ddpm=None, encoder_input="x"): # not used
    labels = np.arange(0, num_classes)
    Y = []
    Y_hat = []
    for batch in loader:
        x, y, sid = batch[:3]
        x, y = x.to(device), y.type(torch.LongTensor).to(device)

        if encoder_input == "x_hat" and ddpm is not None:
            x_hat, *_ = ddpm(x)
            encoder_in = x_hat.detach()
        else:
            encoder_in = x

        encoder_out = encoder(encoder_in)

        y_hat = fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)
        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

    # Convert to numpy arrays
    Y = torch.cat(Y, dim=0).numpy()
    Y_hat = torch.cat(Y_hat, dim=0).numpy() 

    # Calculate metrics
    if task == "P300":
        y_pred = Y_hat.argmax(axis=1)
        accuracy = accuracy_score(Y, y_pred)
        auc = roc_auc_score(Y, Y_hat[:, 1], average="macro")
    else:
        accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
        auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)
    f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    
    metrics = {"accuracy": accuracy,  "f1": f1, "recall": recall, 
               "precision": precision, "auc": auc}
    return metrics

def evaluate_with_subjectwise_znorm(diffe, loader, device, name="Test", num_sessions=6, 
                                    unseen=False, z_stats_train=None, ddpm=None, encoder_input="x"):
    diffe.eval()
    if task == "SSVEP":
        labels = np.arange(0, 26)
    elif task == "MI":
        labels = np.arange(0, 4)
    elif task == "P300":
        labels = np.arange(0, 2)
    else:
        print(f"Warning: Unknown task config '{task}'. Defaulting to 'SSVEP'")
        labels = np.arange(0, 26) 
    Y, Y_hat = [], []

    with torch.no_grad():
        if unseen:
            if task == "P300" and z_local_norm_mode == "option2":
                z_by_sid_sess = {}
                for x_, y_, sid_batch, sess_batch in loader:
                    x_ = x_.to(device)
                    sess_batch = sess_batch.to(device)
                    if encoder_input == "x_hat" and ddpm is not None:
                        x_hat, *_ = ddpm(x_)
                        encoder_in_ = x_hat.detach()
                    elif encoder_input == "x":
                        encoder_in_ = x_
                    else:
                        encoder_in_ = x_

                    z_batch = diffe.encoder(encoder_in_)[1]
                    for i in range(z_batch.size(0)):
                        key = (int(sid_batch[i]), int(sess_batch[i]))
                        z_by_sid_sess.setdefault(key, []).append(z_batch[i].unsqueeze(0))

            all_x, all_y, all_sid = [], [], []
            for batch in loader:
                x, y, sid = batch[:3]
                all_x.append(x)
                all_y.append(y)
                all_sid.append(sid)
            all_x = torch.cat(all_x, dim=0).to(device)
            all_y = torch.cat(all_y, dim=0).to(device)
            all_sid = torch.cat(all_sid, dim=0)

            subjects = all_sid.unique(sorted=True)
            for s in subjects:
                indices = (all_sid == s)
                x_sub = all_x[indices]
                y_sub = all_y[indices]

                # Generate DDPM output if needed
                if encoder_input == "x_hat" and ddpm is not None:
                    x_hat, down, up, noise, t = ddpm(x_sub)
                    encoder_in = x_hat.detach()
                    ddpm_out = (x_hat, down, up, t)
                elif encoder_input == "x" and ddpm is not None:
                    x_hat, down, up, noise, t = ddpm(x_sub)
                    ddpm_out = (x_hat, down, up, t)
                    encoder_in = x_sub
                elif ddpm is None:
                    encoder_in = x_sub
                    ddpm_out = (None, None, None, None)
                    x_hat = None

                # Get embeddings for z-normalization
                z = diffe.encoder(encoder_in)[1]

                if task == "SSVEP": 
                    # Apply z-normalization
                    z_mean = z[:104].mean(dim=0, keepdim=True)
                    z_std = z[:104].std(dim=0, keepdim=True) + 1e-6
                    z_norm = (z - z_mean) / z_std
                elif task == "MI":
                    samples_per_subject = 576  # 2 sessions × 288 samples
                    half = samples_per_subject // 2  # 288
                    quarter = half // 2  # 144

                    # Use only half from each session for stats
                    z_sess0_half = z[:half][:quarter]
                    z_sess1_half = z[half:][:quarter]

                    z_mean0 = z_sess0_half.mean(dim=0, keepdim=True)
                    z_std0 = z_sess0_half.std(dim=0, keepdim=True) + 1e-6

                    z_mean1 = z_sess1_half.mean(dim=0, keepdim=True)
                    z_std1 = z_sess1_half.std(dim=0, keepdim=True) + 1e-6

                    # Average stats
                    avg_mean = (z_mean0 + z_mean1) / 2
                    avg_std = (z_std0 + z_std1) / 2

                    z_norm = (z - avg_mean) / avg_std
                elif task == "P300" and z_local_norm_mode == "option1":
                    samples_per_subject = z.shape[0]
                    half = samples_per_subject // 2  # use a half for calculating z-stats

                    z_half = z[:half]  
                    z_mean = z_half.mean(dim=0, keepdim=True)
                    z_std = z_half.std(dim=0, keepdim=True) + 1e-6

                    z_norm = (z - z_mean) / z_std  
                elif task == "P300" and z_local_norm_mode == "option2":
                    sess_means, sess_stds, z_all = [], [], []
                    for sess_id in range(3):
                        key = (int(s.item()), sess_id)
                        if key in z_by_sid_sess:
                            z_cat = torch.cat(z_by_sid_sess[key])
                            z_all.append(z_cat)
                            z_half = z_cat[: z_cat.size(0) // 2]
                            sess_means.append(z_half.mean(0, keepdim=True))
                            sess_stds.append(z_half.std(0, keepdim=True) + 1e-6)
                    if not sess_means:
                        print(f"[Warning] No session for subject {s.item()}")
                        continue
                    avg_mean = torch.stack(sess_means).mean(0)
                    avg_std = torch.stack(sess_stds).mean(0)
                    z_norm = (torch.cat(z_all) - avg_mean) / avg_std
                else:
                    print(f"Warning: Unknown task config '{task}'. Defaulting to 'SSVEP'")
                    z_mean = z[:104].mean(dim=0, keepdim=True)
                    z_std = z[:104].std(dim=0, keepdim=True) + 1e-6
                    z_norm = (z - z_mean) / z_std

                # Choose appropriate input based on classifier_input setting
                if classifier_input == "z":
                    y_hat = F.softmax(diffe.fc(z_norm), dim=1)

                elif classifier_input == "x":
                    y_hat = F.softmax(diffe.fc(x_sub), dim=1)
                        
                elif classifier_input == "x_hat" and ddpm is not None:
                    y_hat = F.softmax(diffe.fc(x_hat.detach()), dim=1)
                        
                elif classifier_input == "decoder_out" and decoder_variant == "use_decoder":
                    decoder_out, _, _ = diffe(x_sub, ddpm_out)
                    y_hat = F.softmax(diffe.fc(decoder_out.detach()), dim=1)
                else:
                    y_hat = F.softmax(diffe.fc(z_norm), dim=1)
                
                Y.append(y_sub.detach().cpu())
                Y_hat.append(y_hat.detach().cpu())
        else:
            # For seen subjects: use provided z_stats_train from training data
            if z_stats_train is None and classifier_input == "z":
                raise ValueError("z_stats_train must be provided for seen subject evaluation with z input.")

            for batch in loader:
                x, y, sid = batch[:3]
                x, y = x.to(device), y.type(torch.LongTensor).to(device)
                #x, y = x.to(device), y.to(device)

                # Generate DDPM output if needed
                if encoder_input == "x_hat" and ddpm is not None:
                    x_hat, down, up, noise, t = ddpm(x)
                    encoder_in = x_hat.detach()
                    ddpm_out = (x_hat, down, up, t)
                elif encoder_input == "x" and ddpm is not None:
                    x_hat, down, up, noise, t = ddpm(x)
                    ddpm_out = (x_hat, down, up, t)
                    encoder_in = x
                elif ddpm is None:
                    encoder_in = x
                    ddpm_out = (None, None, None, None)
                    x_hat = None

                # Get embeddings and apply z-normalization
                _, z = diffe.encoder(encoder_in)
                
                # Apply subject-wise z-normalization using training statistics
                z_norm = torch.stack([
                    (z[i] - z_stats_train[int(sid[i].item())][0].squeeze(0)) /
                    z_stats_train[int(sid[i].item())][1].squeeze(0)
                    for i in range(z.size(0))])

                # Choose appropriate input based on classifier_input setting
                if classifier_input == "z":
                    y_hat = F.softmax(diffe.fc(z_norm), dim=1)

                elif classifier_input == "x":
                    y_hat = F.softmax(diffe.fc(x), dim=1)
                        
                elif classifier_input == "x_hat" and ddpm is not None:
                    y_hat = F.softmax(diffe.fc(x_hat.detach()), dim=1)
                        
                elif classifier_input == "decoder_out" and decoder_variant == "use_decoder":
                    decoder_out, _, _ = diffe(x, ddpm_out)
                    y_hat = F.softmax(diffe.fc(decoder_out.detach()), dim=1)
                else:
                    y_hat = F.softmax(diffe.fc(z_norm), dim=1)

                Y.append(y.detach().cpu())
                Y_hat.append(y_hat.detach().cpu())

    Y = torch.cat(Y).numpy()
    Y_hat = torch.cat(Y_hat).numpy()

    # Calculate metrics (unchanged)
    if task == "P300":
        y_pred = Y_hat.argmax(axis=1)
        accuracy = accuracy_score(Y, y_pred)
        auc = roc_auc_score(Y, Y_hat[:, 1], average="macro")
    else:
        accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
        auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)
    f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
    
    metrics = {"accuracy": accuracy, "f1": f1, "recall": recall, 
               "precision": precision, "auc": auc}
    return metrics

def initialize_models():
    # DDPM model
    if ddpm_variant == "use_ddpm":
        ddpm_model = ConditionalUNet(in_channels=channels, n_feat=ddpm_dim).to(device)
        ddpm = DDPM(nn_model=ddpm_model, betas=(1e-6, 1e-2), n_T=n_T, device=device).to(device)
    else:
        ddpm = None
    
    # Encoder 
    encoder = EEGNet(nb_classes=num_classes, 
                     Chans=channels, 
                     Samples=timepoints, 
                     dropoutRate=eegnet_params["dropout_rate"],
                     kernLength=eegnet_params["kernel_length"], 
                     F1=eegnet_params["F1"], 
                     D=eegnet_params["D"], 
                     F2=eegnet_params["F2"],
                     dropoutType=eegnet_params["dropout_type"]).to(device)
    
    # Decoder
    if decoder_variant == "use_decoder":
        decoder = Decoder(in_channels=channels, n_feat=ddpm_dim, encoder_dim=encoder_dim).to(device)
    else:
        decoder = None
    
    # Classifier
    if classifier_variant == "eegnet_classifier":
        fc = EEGNetClassifier(nb_classes=eegnet_classifier_params["nb_classes"],
                              Chans=eegnet_classifier_params["Chans"],
                              Samples=eegnet_classifier_params["Samples"],
                              dropoutRate=eegnet_classifier_params["dropoutRate"],
                              kernLength=eegnet_classifier_params["kernLength"],
                              F1=eegnet_classifier_params["F1"],
                              D=eegnet_classifier_params["D"],
                              F2=eegnet_classifier_params["F2"],
                              dropoutType=eegnet_classifier_params["dropoutType"]).to(device)
    elif classifier_variant == "fc_classifier":
        fc = LinearClassifier(encoder_dim, fc_dim, emb_dim=num_classes).to(device)
    else:
        raise ValueError(f"Unknown classifier variant: {classifier_variant}")
    
    # DiffE combines everything
    diffe = DiffE(encoder, decoder, fc).to(device)
    
    # Print model summary
    print("\n--------- Model Summary ---------")
    print("Input channels    :", channels)
    print("Timepoints        :", timepoints)
    
    # Print DDPM parameters if it exists
    if ddpm_variant == "use_ddpm" and ddpm is not None:
        print("DDPM parameters   :", sum(p.numel() for p in ddpm.parameters()))
    else:
        print("DDPM parameters   : 0 (no DDPM used)")
    
    print("Encoder parameters:", sum(p.numel() for p in encoder.parameters()))
    
    # Print decoder parameters if it exists
    if decoder_variant == "use_decoder" and decoder is not None:
        print("Decoder parameters:", sum(p.numel() for p in decoder.parameters()))
    else:
        print("Decoder parameters: 0 (no decoder used)")
    
    print("Classifier params :", sum(p.numel() for p in fc.parameters()))
    print("Total DiffE params:", sum(p.numel() for p in diffe.parameters()))
    print("-------------------------------\n")
    
    return ddpm, diffe

def setup_optimizers(ddpm, diffe):
    # Optimizers
    if ddpm_variant == "use_ddpm" and ddpm is not None:
        optim1 = optim.RMSprop(ddpm.parameters(), lr=base_lr)
        scheduler1 = optim.lr_scheduler.CyclicLR(optimizer=optim1, 
                                             base_lr=base_lr,
                                             max_lr=max_lr, 
                                             step_size_up=scheduler_step_size,
                                             mode="exp_range", 
                                             cycle_momentum=False,
                                             gamma=scheduler_gamma)
    else:
        # Create dummy optimizer and scheduler when DDPM is not used
        optim1 = None
        scheduler1 = None
    
    optim2 = optim.RMSprop(diffe.parameters(), lr=base_lr)
    
    # EMA
    fc_ema = EMA(diffe.fc, 
                 beta=ema_beta, 
                 update_after_step=ema_update_after, 
                 update_every=ema_update_every)
    
    # Learning rate scheduler for DiffE
    scheduler2 = optim.lr_scheduler.CyclicLR(optimizer=optim2, 
                                         base_lr=base_lr,
                                         max_lr=max_lr, 
                                         step_size_up=scheduler_step_size,
                                         mode="exp_range", 
                                         cycle_momentum=False,
                                         gamma=scheduler_gamma)
    
    return optim1, optim2, fc_ema, scheduler1, scheduler2

def train_epoch(ddpm, diffe, train_loader, optim1, optim2, scheduler1, scheduler2, fc_ema, epoch, z_stats, proj_head, supcon_loss):

    # Only put DDPM in train mode if it exists
    if ddpm_variant == "use_ddpm" and ddpm is not None:
        ddpm.train()
    diffe.train()
    
    # Initialize tracking variables
    epoch_loss = 0
    num_batches = 0
    epoch_acc = 0
    total_samples = 0

    for batch in train_loader:
        x, y, sid = batch[:3]
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        y_cat = F.one_hot(y, num_classes=num_classes).type(torch.FloatTensor).to(device)

        # Train DDPM
        if ddpm_variant == "use_ddpm" and ddpm is not None and optim1 is not None:
            optim1.zero_grad()
            x_hat, down, up, noise, t = ddpm(x)

            # Align dimensions if needed
            if x_hat.shape[-1] != x.shape[-1]:
                target_len = min(x_hat.shape[-1], x.shape[-1])
                x_hat = F.interpolate(x_hat, size=target_len)
                x = F.interpolate(x, size=target_len)

            loss_ddpm = F.l1_loss(x_hat, x, reduction="none")
            loss_ddpm.mean().backward()
            optim1.step()
            ddpm_out = x_hat, down, up, t
        else:
            x_hat = None
            ddpm_out = None
      
        # Train DiffE
        optim2.zero_grad()

        if encoder_input == "x_hat" and ddpm_variant != "use_ddpm":
            encoder_in = x
        else:
            encoder_in = x_hat.detach() if encoder_input == "x_hat" else x

        if decoder_variant == "no_decoder":
            _, fc_out, z = diffe(encoder_in, ddpm_out)
            loss_decoder = 0.0
        else:
            decoder_out, fc_out, z = diffe(encoder_in, ddpm_out)
            if ddpm_variant == "use_ddpm":
                loss_decoder = F.l1_loss(decoder_out, x_hat.detach()) # we detached x_hat, cause we don't want to backprop through the DDPM
            else:
                if decoder_out.shape[-1] != x.shape[-1]:
                    target_len = min(decoder_out.shape[-1], x.shape[-1])
                    decoder_out = F.interpolate(decoder_out, size=target_len)
                    x = F.interpolate(x, size=target_len)
                loss_decoder = F.l1_loss(decoder_out, x)

        # Normalize by subject
        if isinstance(use_subject_wise_z_norm, dict) and use_subject_wise_z_norm.get("train", True):
            z = torch.stack([(z[i] - z_stats[int(sid[i].item())][0].squeeze(0)) / 
                z_stats[int(sid[i].item())][1].squeeze(0) 
                for i in range(z.size(0))])
        
        # Losses 
        # --- Classification loss
        if classification_loss == "CE":
            loss_c = nn.CrossEntropyLoss()(fc_out, y)
        elif classification_loss == "MSE":
            loss_c = nn.MSELoss()(fc_out, y_cat)
        else:
            raise ValueError(f"Unknown classification loss: {classification_loss}")

        # --- Contrastive loss
        if contrastive_loss == "SupCon":
            z_proj = proj_head(z)
            loss_supcon = supcon_loss(z_proj, y)
        else:
            loss_supcon = 0.0 
        
        # --- Combined loss
        # Loss weights (scheduler logic)
        alpha_val = alpha  # always float
        if isinstance(beta, str) and beta.startswith("scheduler"):
            beta_val = min(1.0, epoch / 50) * beta_scale
        else:
            beta_val = float(beta)
        if isinstance(gamma, str) and gamma.startswith("scheduler"):
            gamma_val = min(1.0, epoch / 100) * gamma_scale
        else:
            gamma_val = float(gamma)

        loss = alpha_val * loss_c + beta_val * loss_supcon + gamma_val * loss_decoder
        loss.backward()
        optim2.step()
        
        # Update schedulers and EMA
        if scheduler1 is not None:
            scheduler1.step()
        scheduler2.step()
        fc_ema.update()
        
        # Track metrics
        epoch_loss += loss.item()
        num_batches += 1
        
        # Calculate accuracy
        pred_labels = torch.argmax(fc_out, dim=1)
        correct = (pred_labels == y).sum().item()
        epoch_acc += correct
        total_samples += y.size(0)
    
    return epoch_loss / num_batches, epoch_acc / total_samples

def validate(ddpm, diffe, val_loader, z_stats, proj_head, supcon_loss, alpha, beta, gamma):

    if ddpm_variant == "use_ddpm" and ddpm is not None:
        ddpm.eval()
    
    diffe.eval()
    
    # Get metrics using the evaluate function
    metrics_val = evaluate_with_subjectwise_znorm(diffe, val_loader, device, name="Val", unseen=False, z_stats_train=z_stats, ddpm=ddpm, encoder_input=encoder_input) # metrics_val = evaluate(diffe.encoder, diffe.fc, val_loader, device, ddpm=ddpm, encoder_input=encoder_input) 

    # Calculate validation loss
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x, y, sid = batch[:3]
            x, y = x.to(device), y.type(torch.LongTensor).to(device)
            y_cat = F.one_hot(y, num_classes=num_classes).float().to(device)
            
            if ddpm_variant == "use_ddpm":
                x_hat, down, up, noise, t = ddpm(x)
                ddpm_out = x_hat, down, up, t

                if x_hat.shape[-1] != x.shape[-1]:
                    target_len = min(x_hat.shape[-1], x.shape[-1])
                    x_hat = F.interpolate(x_hat, size=target_len)
                    x = F.interpolate(x, size=target_len)

                loss_ddpm = F.l1_loss(x_hat, x, reduction="none") 
            else:
                x_hat = None
                ddpm_out = None

            if encoder_input == "x_hat" and ddpm_variant != "use_ddpm":
                encoder_in = x
            else:
                encoder_in = x_hat.detach() if encoder_input == "x_hat" else x

            if decoder_variant == "no_decoder":
                _, fc_out, z = diffe(encoder_in, ddpm_out)
                loss_decoder = 0.0
            else:
                decoder_out, fc_out, z = diffe(encoder_in, ddpm_out)
                if ddpm_variant == "use_ddpm":
                    loss_decoder = F.l1_loss(decoder_out, x_hat.detach())
                else:
                    if decoder_out.shape[-1] != x.shape[-1]:
                        target_len = min(decoder_out.shape[-1], x.shape[-1])
                        decoder_out = F.interpolate(decoder_out, size=target_len)
                        x = F.interpolate(x, size=target_len)
                    loss_decoder = F.l1_loss(decoder_out, x)
            
            
            if isinstance(use_subject_wise_z_norm, dict) and use_subject_wise_z_norm.get("train", True):
                z = torch.stack([(z[i] - z_stats[int(sid[i].item())][0].squeeze(0)) / 
                    z_stats[int(sid[i].item())][1].squeeze(0) 
                    for i in range(z.size(0))])

            # --- Classification loss
            if classification_loss == "CE":
                loss_c = nn.CrossEntropyLoss()(fc_out, y)
            elif classification_loss == "MSE":
                loss_c = nn.MSELoss()(fc_out, y_cat)
            else:
                raise ValueError(f"Unknown classification loss: {classification_loss}")
            
            # --- Contrastive loss
            if contrastive_loss == "SupCon":
                z_proj = proj_head(z)
                loss_supcon = supcon_loss(z_proj, y)
            else:
                loss_supcon = 0.0  # Add more options if needed
            
            # --- Combined loss
            val_loss += (alpha * loss_c + beta * loss_supcon + gamma * loss_decoder).item()
    
    val_loss = val_loss / len(val_loader)
    return metrics_val, val_loss

def train():

    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Setup data loaders
    if task == "SSVEP":
        loaders = load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed) 
    elif task == "MI":
        loaders = MI_load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed)
    elif task == "P300":
        loaders = P300_load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed,num_workers=num_workers, pin_memory=pin_memory)  
    else:
        print(f"Warning: Unknown task config '{task}'. Defaulting to 'SSVEP'")
        loaders = load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed) 
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    
    # Initialize models
    ddpm, diffe = initialize_models()
    
    # Setup optimizers and schedulers
    optim1, optim2, fc_ema, scheduler1, scheduler2 = setup_optimizers(ddpm, diffe)
    
    # Setup training auxiliaries
    z_stats = get_subjectwise_z_stats(train_loader, diffe.encoder, device)
    supcon_loss = SupConLoss(temperature=supcon_temperature)
    proj_head = ProjectionHead(input_dim=encoder_dim, proj_dim=128).to(device)
    
    # Initialize tracking variables
    best_metrics = {"acc": 0, "f1": 0, "recall": 0, "precision": 0, "auc": 0, "epoch": 0, "model_path": None}
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "timestamps": []}
    val_acc = 0.0

    # Training loop
    start_time = time.time()
    with tqdm(total=num_epochs, desc=f"Training") as pbar:
        for epoch in range(num_epochs):

            epoch_start = time.time()
            
            # Train for one epoch
            train_loss, train_acc = train_epoch(ddpm, diffe, train_loader, optim1, optim2, scheduler1, scheduler2,
                                                fc_ema, epoch, z_stats, proj_head, supcon_loss)
            
            # Record training metrics
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            # Validate model
            # Loss weights (scheduler logic)
            if isinstance(beta, str) and beta.startswith("scheduler"):
                beta_val = min(1.0, epoch / 50) * beta_scale
            else:
                beta_val = float(beta)
            if isinstance(gamma, str) and gamma.startswith("scheduler"):
                gamma_val = min(1.0, epoch / 100) * gamma_scale
            else:
                gamma_val = float(gamma)
            
            # Run validation at appropriate intervals
            if epoch > start_test and epoch % test_frequency == 0:
                metrics_val, val_loss = validate(ddpm, diffe, val_loader, z_stats, proj_head, 
                                                 supcon_loss, alpha, beta_val, gamma_val)
                
                # Record validation metrics
                val_acc = metrics_val["accuracy"]
                f1 = metrics_val["f1"]
                recall = metrics_val["recall"]
                precision = metrics_val["precision"]
                auc = metrics_val["auc"]
                
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                
                # Check for best metrics
                if val_acc > best_metrics["acc"]:
                    best_metrics["acc"] = val_acc
                    best_metrics["epoch"] = epoch
                    # Save model
                    model_path = os.path.join(checkpoints_dir, f"diffe_best_acc.pth")
                    torch.save(diffe.state_dict(), model_path)
                    best_metrics["model_path"] = model_path
                
                if f1 > best_metrics["f1"]:
                    best_metrics["f1"] = f1
                if recall > best_metrics["recall"]:
                    best_metrics["recall"] = recall
                if precision > best_metrics["precision"]:
                    best_metrics["precision"] = precision
                if auc > best_metrics["auc"]:
                    best_metrics["auc"] = auc
                
                # Update progress bar
                description = f"Val Acc: {val_acc*100:.2f}% | Best: {best_metrics['acc']*100:.2f}%"
                pbar.set_description(description)
            
            # Track time
            epoch_time = time.time() - epoch_start
            history["timestamps"].append(epoch_time)
            
            # Print epoch summary
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Train Acc: {train_acc*100:.2f}% | "
                  f"Val Acc: {val_acc*100:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            # Update progress bar
            pbar.update(1)
    
    # Final evaluation
    total_time = time.time() - start_time
    print(f"\n===== Training completed in {total_time/60:.2f} minutes =====")
    print(f"Best validation accuracy: {best_metrics['acc']*100:.2f}% at epoch {best_metrics['epoch']}")
    print(f"Best F1 score: {best_metrics['f1']*100:.2f}%")
    
    # Save final model
    final_model_path = os.path.join(checkpoints_dir, f"diffe_final.pth")
    torch.save(diffe.state_dict(), final_model_path)
    
    # Save history
    history_df = pd.DataFrame({
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'] if len(history['val_loss']) == num_epochs else history['val_loss'] + [float('nan')] * (num_epochs - len(history['val_loss'])),
        'val_acc': history['val_acc'] if len(history['val_acc']) == num_epochs else history['val_acc'] + [float('nan')] * (num_epochs - len(history['val_acc'])),
        'epoch_time': history['timestamps']})
    history_df.to_csv(os.path.join(log_dir, 'training_history.csv'), index=False)
    
    # Return best model
    return best_metrics, z_stats

def test_best_model(best_metrics, z_stats_train):

    # Load best model
    ddpm, diffe = initialize_models()

    if best_metrics["model_path"] is not None:
        try:
            diffe.load_state_dict(torch.load(best_metrics["model_path"]))
            print(f"Loaded best model from {best_metrics['model_path']}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using the final model state instead.")
    else:
        print("No best model was saved (validation accuracy didn't improve). Using final model state.")
    
    # Load test data
    if task == "SSVEP":
        loaders = load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed)
    elif task == "MI":
        loaders = MI_load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed)
    elif task == "P300":
        loaders = P300_load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed,num_workers=num_workers, pin_memory=pin_memory)
    else:
        print(f"Warning: Unknown task config '{task}'. Defaulting to 'SSVEP'")
        loaders = load_split_dataset(root_dir=data_dir, num_seen=num_seen, seed=seed) 

    
    test1_loader = loaders["test1"]
    test2_loader = loaders["test2"]
    
    # Evaluate on test sets
    diffe.eval()

    # Determine which normalization strategy to use based on config
    z_norm_mode = use_subject_wise_z_norm.get("mode", "option2")
    print(f"Using Z-normalization mode: {z_norm_mode}")

    if z_norm_mode == "option1": # Option 1: Z-norm in train only; standard test eval
        test1_metrics = evaluate(diffe.encoder, diffe.fc, test1_loader, device, ddpm=ddpm, encoder_input=encoder_input)
        test2_metrics = evaluate(diffe.encoder, diffe.fc, test2_loader, device, ddpm=ddpm, encoder_input=encoder_input)
    
    elif z_norm_mode == "option2": # Option 2: Z-norm in train + test; test_seen uses train stats, test_unseen uses calibration
        test1_metrics = evaluate_with_subjectwise_znorm(diffe, test1_loader, device, name="Test1", unseen=False, z_stats_train=z_stats_train, ddpm=ddpm, encoder_input=encoder_input)
        test2_metrics = evaluate_with_subjectwise_znorm(diffe, test2_loader, device, name="Test2", unseen=True, ddpm=ddpm, encoder_input=encoder_input)
    
    elif z_norm_mode == "option3": # Option 3: Standard test_seen; test_unseen uses calibration
        test1_metrics = evaluate(diffe.encoder, diffe.fc, test1_loader, device, ddpm=ddpm, encoder_input=encoder_input)
        test2_metrics = evaluate_with_subjectwise_znorm(diffe, test2_loader, device, name="Test2", unseen=True, ddpm=ddpm, encoder_input=encoder_input)
    else:
        print(f"Unknown Z-normalization mode: {z_norm_mode}. Using default evaluation.")
        test1_metrics = evaluate(diffe.encoder, diffe.fc, test1_loader, device, ddpm=ddpm, encoder_input=encoder_input)
        test2_metrics = evaluate(diffe.encoder, diffe.fc, test2_loader, device, ddpm=ddpm, encoder_input=encoder_input)

    print("\n===== Test Results =====")
    print(f"Test1 accuracy: {test1_metrics['accuracy']*100:.2f}%")
    print(f"Test1 F1 score: {test1_metrics['f1']*100:.2f}%")
    print(f"Test2 accuracy: {test2_metrics['accuracy']*100:.2f}%")
    print(f"Test2 F1 score: {test2_metrics['f1']*100:.2f}%")
    
    # Save test results
    results = {"test1": test1_metrics, 
               "test2": test2_metrics,
               "best_val": best_metrics}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.save(os.path.join(log_dir, f'test_results_{timestamp}.npy'), results)
    
    return results, z_stats_train

if __name__ == "__main__":

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    print(f"Using device: {device}")
    
    # Run training
    best_metrics, z_stats_train = train()

    # Load training history from the saved CSV
    try:
        history_df = pd.read_csv(os.path.join(log_dir, 'training_history.csv'))
        history = {'train_loss': history_df['train_loss'].tolist(),
                   'train_acc': history_df['train_acc'].tolist(),
                   'val_loss': history_df['val_loss'].dropna().tolist(),
                   'val_acc': history_df['val_acc'].dropna().tolist()}
        # Plot training progress
        plot_training_progress(history, log_dir)
    except Exception as e:
        print(f"Could not plot training progress: {e}")
    
    # Test best model
    test_results = test_best_model(best_metrics, z_stats_train)