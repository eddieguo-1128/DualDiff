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

# --------- Command-line arguments (optional) ---------

# --------- Work directory  ---------
option = "drive"  # "local" or "drive"
if option == "local":
    work_dir = "/Users/kshapovalenko/Desktop/GITHUB/DualDiff-LOCAL"
elif option == "drive":
    work_dir = "/content/drive/MyDrive/Communikate/IDL-research/"

# --------- Reproducibility  ---------
seed = 44
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------- Dataset  ---------
if option == "local":
    data_dir = os.path.join(work_dir, "SSVEP-CHAR")
elif option == "drive":
    data_dir = "/content/drive/MyDrive/Communikate//IDL-research/dataset/ssvep/chars/"
num_subjects = 35

# --------- Logging  ---------
run_name = "run3" 
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
num_classes = 26
channels = 64 
timepoints = 250  # From EEGNet parameters

# DDPM parameters
n_T = 1000
ddpm_dim = 128
encoder_dim = 256
fc_dim = 512

# --------- Training hyperparams ---------
num_epochs = 500
batch_size = 32
batch_size_eval = 260
test_period = 1
start_test = test_period

# Optimizer settings
base_lr = 9e-5
max_lr = 1.5e-3
scheduler_step_size = 150
scheduler_gamma = 0.9998

# Loss weights
initial_alpha = 1.0
beta_scale = 0.2  # Multiplied by min(1.0, epoch/50)
gamma_scale = 0.05  # Multiplied by min(1.0, epoch/100)

# EMA parameters
ema_beta = 0.95
ema_update_after = 100
ema_update_every = 10

# SupCon loss parameters
supcon_temperature = 0.07

# Model-specific parameters
eegnet_params = {"dropout_rate": 0.2, "kernel_length": 64,
                 "F1": 16, "D": 2, "F2": 32, "dropout_type": "Dropout"}

# --------- Evaluation ---------
test_frequency = 1

# --------- Testing ---------
use_subject_wise_z_norm = {
    "train": True,               # normalize z using subject-wise stats during training
    "test_seen": "train",        # normalize test_seen using training stats (from train_loader)
    "test_unseen": "calibrate"   # normalize test_unseen using 0–3 session stats (from test data)
}