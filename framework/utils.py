import torch
import torch.nn as nn
from config import *

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
        if task == "SSVEP":
            #print(f"[Check] Subject {sid}: z samples = {z_cat.shape[0]} (expected: {26 * 4})")
            assert z_cat.shape[0] >= 104, f"Subject {sid} z count < 104 — check loader input!"
            z_4session = z_cat[:26 * 4]  # only take the first 104 
            z_mean = z_4session.mean(dim=0, keepdim=True)
            z_std = z_4session.std(dim=0, keepdim=True) + 1e-6
            z_stats[sid] = (z_mean, z_std)
        elif task == "MI":
            n = z_cat.shape[0]
            half = n // 2

            # session 0 and session 1 split
            z_sess0 = z_cat[:half]
            z_sess1 = z_cat[half:]

            # per-session stats
            z_mean0 = z_sess0.mean(dim=0, keepdim=True)
            z_std0 = z_sess0.std(dim=0, keepdim=True) + 1e-6

            z_mean1 = z_sess1.mean(dim=0, keepdim=True)
            z_std1 = z_sess1.std(dim=0, keepdim=True) + 1e-6

            # average stats across sessions
            avg_mean = (z_mean0 + z_mean1) / 2
            avg_std = (z_std0 + z_std1) / 2

            print(f"[Check] Subject {sid}: total={n}, sess0={half}, sess1={n-half}")
            z_stats[sid] = (avg_mean, avg_std)
        else:
            print(f"Warning: Unknown task config '{task}'. Defaulting to 'SSVEP'")
            #print(f"[Check] Subject {sid}: z samples = {z_cat.shape[0]} (expected: {26 * 4})")
            assert z_cat.shape[0] >= 104, f"Subject {sid} z count < 104 — check loader input!"
            z_4session = z_cat[:26 * 4]  # only take the first 104 
            z_mean = z_4session.mean(dim=0, keepdim=True)
            z_std = z_4session.std(dim=0, keepdim=True) + 1e-6
            z_stats[sid] = (z_mean, z_std)

    return z_stats
