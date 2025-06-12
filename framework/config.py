import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ema_pytorch import EMA
from tqdm import tqdm
from sklearn.metrics import (f1_score, roc_auc_score, precision_score, recall_score, top_k_accuracy_score)
import time
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       module="sklearn.metrics._classification")
warnings.filterwarnings("ignore", message="This filename .* does not conform to MNE naming conventions.*",
                        category=RuntimeWarning, module="mne.io")

# --------- Command-line arguments (optional) ---------

task = "MI" #"SSVEP","MI" or "FEIS"

# --------- Work directory  ---------
option = "drive"  # "local" or "drive"
if option == "local":
    work_dir = "/Users/kshapovalenko/Desktop/GITHUB/DualDiff-LOCAL"
elif option == "drive":
    work_dir = "/content/drive/MyDrive/project/model/MI/sweep4-new"  

# --------- Reproducibility  ---------
seed = int(os.environ.get("SEED", "44"))
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Dataset  ---------
if option == "local":
    data_dir = os.path.join(work_dir, "cleaned_data")
    label_dir = os.path.join(work_dir, "second_session_labels")
elif option == "drive":
    data_dir = "/content/drive/MyDrive/project/dataset/MI/cleaned_data"
    label_dir = "/content/drive/MyDrive/project/dataset/MI/second_session_labels"
num_subjects = 9 #SSVEP:35; MI:9
num_seen = 7 #SSVEP:35; MI:9

# --------- Logging  ---------
run_name = os.environ.get("RUN_NAME", "run1")
## Run directory 
run_dir = os.path.join(work_dir, run_name)
os.makedirs(run_dir, exist_ok=True)
## Log directory
log_dir = os.path.join(run_dir, "logs")
checkpoints_dir = os.path.join(run_dir, "checkpoints")

## Weights & Biases (optional)
wandb_project = "DualDiff"
wandb_run_name = run_name

# --------- Model ---------
num_classes = 4 #SSVEP:26; MI:4
channels = 22 #SSVEP:64; MI:22
timepoints = 1001  # From EEGNet parameters. SSVEP:250; MI:1001 

# DDPM 
ddpm_variant = os.environ.get("DDPM_VARIANT", "use_ddpm")  # "use_ddpm" or "no_ddpm"
n_T = 1000
ddpm_dim = 128
encoder_dim = 256
fc_dim = 512

# Encoder parameters
encoder_input = os.environ.get("ENCODER_INPUT", "x")       # "x" or "x_hat"
eegnet_params = {"dropout_rate": 0.2, "kernel_length": 64,
                 "F1": 16, "D": 2, "F2": 32, "dropout_type": "Dropout"}

# Decoder parameters
decoder_variant = os.environ.get("DECODER_VARIANT", "use_decoder")  # "use_decoder" or "no_decoder"
decoder_input = os.environ.get("DECODER_INPUT", "z only") # Choose from: 
                                                            ## "x + x_hat + skips"
                                                            ## "x + x_hat"
                                                            ## "x_hat + skips"
                                                            ## "x + skips"
                                                            ## "skips"
                                                            ## "z only"
                                                            ## "z + x"
                                                            ## "z + x_hat"
                                                            ## "z + skips"

# Classifier parameters
classifier_variant = os.environ.get("CLASSIFIER_VARIANT", "fc_classifier")  # "eegnet_classifier" or "fc_classifier"
classifier_input = os.environ.get("CLASSIFIER_INPUT", "z")  # "x", "x_hat", "decoder_out", "input_mixup", or "z"
eegnet_classifier_params = {"nb_classes": num_classes,
                            "Chans": channels,
                            "Samples": timepoints,
                            "dropoutRate": eegnet_params["dropout_rate"],
                            "kernLength": eegnet_params["kernel_length"],
                            "F1": eegnet_params["F1"], 
                            "D": eegnet_params["D"], 
                            "F2": eegnet_params["F2"],
                            "dropoutType": eegnet_params["dropout_type"]}

# --------- Training hyperparams ---------
num_epochs = 500 # for all ablations, do 500 epochs
batch_size = 32
batch_size_eval = 260
test_period = 1
start_test = test_period

# Optimizer settings
base_lr = 9e-5
max_lr = 1.5e-3
scheduler_step_size = 150
scheduler_gamma = 0.9998

# Loss 
ddpm_reconstruction_loss = os.environ.get("DDPM_RECONSTRUCTION_LOSS", "True") == "True"
classification_loss = os.environ.get("CLASSIFICATION_LOSS", "CE")
contrastive_loss = os.environ.get("CONTRASTIVE_LOSS", "SupCon")
decoder_reconstruction_loss = os.environ.get("DECODER_RECONSTRUCTION_LOSS", "L1")

alpha = float(os.environ.get("ALPHA", 1.0))
beta = os.environ.get("BETA", "scheduler to 0.05")
gamma = os.environ.get("GAMMA", "scheduler to 0.2")

beta_scale = 0.2  # Multiplied by min(1.0, epoch/50)
gamma_scale = 0.05  # Multiplied by min(1.0, epoch/100)

# EMA parameters
ema_beta = 0.95
ema_update_after = 100
ema_update_every = 10

# SupCon loss parameters
supcon_temperature = 0.07

# --------- Evaluation ---------
test_frequency = 1

# --------- Testing ---------
use_subject_wise_z_norm = {"mode": os.environ.get("Z_NORM_MODE", "option2"), "train": True} # Choose from:
                                            # "option1": Z-norm in train only; standard test eval
                                            # "option2": Z-norm in train + test; test_seen uses train stats, test_unseen uses calibration
                                            # "option3": Standard test_seen; test_unseen uses calibration
                                            # "none"