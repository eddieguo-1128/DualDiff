from models import *
from utils import *

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
    top_k_accuracy_score,
)

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

# load the data loader from the file
args = parser.parse_args()
# load the pre-trained model from the file
diffe.load_state_dict(torch.load(args.model_path))

# load the data loader from the file
with open(args.data_loader_path, 'rb') as f:
    data_loader = pickle.load(f)

# Inference loop
diffe.eval()
with torch.no_grad():
    labels = np.arange(0, num_classes)
    Y = []
    Y_hat = []
    for x, y, sid in data_loader:
        x, y = x.to(device), y.type(torch.LongTensor).to(device)
        encoder_out = diffe.encoder(x)
        y_hat = fc(encoder_out[1])
        y_hat = F.softmax(y_hat, dim=1)

        Y.append(y.detach().cpu())
        Y_hat.append(y_hat.detach().cpu())

Y = torch.cat(Y, dim=0).numpy()
Y_hat = torch.cat(Y_hat, dim=0).numpy()

accuracy = top_k_accuracy_score(Y, Y_hat, k=1, labels=labels)
f1 = f1_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
recall = recall_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
precision = precision_score(Y, Y_hat.argmax(axis=1), average="macro", labels=labels)
auc = roc_auc_score(Y, Y_hat, average="macro", multi_class="ovo", labels=labels)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1: {f1:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"AUC: {auc:.4f}")