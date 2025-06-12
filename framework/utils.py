import torch
import torch.nn as nn
from config import *
z_local_norm_mode = os.environ.get("Z_LOCAL_NORM_MODE", "option1")

def get_subjectwise_z_stats(loader, encoder, device, num_sessions=6):
    """
    Compute z_mean and z_std for each subject using only sessions 0–3.

    Returns:
        A dictionary: {subject_id: (mean, std)} where each value has shape [1, D]

    Also prints whether each subject used exactly 104 samples
    (i.e., 4 sessions * 26 samples/session).
    """
    encoder.eval()

    with torch.no_grad():
        if task == "P300" and z_local_norm_mode == "option2":
            z_by_sid_sess = {}
            for x, y, sid_batch,sess_batch in loader:
                x = x.to(device)
            _, z_batch = encoder(x)  # shape: [B, D]
            for i in range(z_batch.size(0)):
                #print(f"[Check] sid: {sid_batch[i]}, sess: {sess_batch[i]}")
                sid = int(sid_batch[i].item())
                sess = int(sess_batch[i].item())
                key = (sid, sess)
                if key not in z_by_sid_sess:
                    z_by_sid_sess[key] = []
                z_by_sid_sess[key].append(z_batch[i].unsqueeze(0))
        else:
            z_by_sid = {}
            for x, y, sid in loader:
                _, z = encoder(x)
                for i in range(z.size(0)):
                    s = int(sid[i].item())
                    if s not in z_by_sid:
                        z_by_sid[s] = []
                    z_by_sid[s].append(z[i].unsqueeze(0))

    z_stats = {}
    if task == "P300" and z_local_norm_mode == "option2":
        subject_ids = set(sid for sid, _ in z_by_sid_sess.keys())
        for sid in sorted(subject_ids):
            sess_means, sess_stds = [], []
            for sess in [0, 1, 2]:
                key = (sid, sess)
                if key in z_by_sid_sess:
                    z_cat = torch.cat(z_by_sid_sess[key], dim=0)  # [N, D]
                    mean = z_cat.mean(dim=0, keepdim=True)
                    std = z_cat.std(dim=0, keepdim=True) + 1e-6
                    sess_means.append(mean)
                    sess_stds.append(std)
            if len(sess_means) > 0:
                avg_mean = torch.stack(sess_means).mean(dim=0)
                avg_std = torch.stack(sess_stds).mean(dim=0)
                z_stats[sid] = (avg_mean, avg_std)
                print(f"[Check] Subject {sid}: {len(sess_means)} sessions used.")
            else:
                print(f"[Warning] No session data found for subject {sid}!")
    else:
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
            elif task == "P300" and z_local_norm_mode == "option1":
                z_mean = z_cat.mean(dim=0, keepdim=True)
                z_std = z_cat.std(dim=0, keepdim=True) + 1e-6
                print(f"[Z STATS] Subject {sid}: {z_cat.shape[0]} samples | mean {z_mean.mean():.4f}, std {z_std.mean():.4f}")
                z_stats[sid] = (z_mean, z_std)         
            else:
                print(f"Warning: Unknown task config '{task}'. Defaulting to 'SSVEP'")
                #print(f"[Check] Subject {sid}: z samples = {z_cat.shape[0]} (expected: {26 * 4})")
                assert z_cat.shape[0] >= 104, f"Subject {sid} z count < 104 — check loader input!"
                z_4session = z_cat[:26 * 4]  # only take the first 104 
                z_mean = z_4session.mean(dim=0, keepdim=True)
                z_std = z_4session.std(dim=0, keepdim=True) + 1e-6
                z_stats[sid] = (z_mean, z_std)

    return z_stats

def compute_z_norm_p300_option2(diffe, loader, device):
    """
    For task=P300 and z_local_norm_mode=option2:
    Compute session-wise z stats and return dict:
        {
            sid: {
                "z_norm": Tensor,        # normalized z across all sessions
                "y": Tensor              # corresponding ground truth
            }
        }
    """
    z_by_sid_sess = {}
    y_by_sid = {}

    for x, y, sid_batch, sess_batch in loader:
        x = x.to(device)
        _, z_batch = diffe.encoder(x)
        for i in range(z_batch.size(0)):
            sid = int(sid_batch[i].item())
            sess = int(sess_batch[i].item())
            key = (sid, sess)
            if key not in z_by_sid_sess:
                z_by_sid_sess[key] = []
            z_by_sid_sess[key].append(z_batch[i].unsqueeze(0))

            if sid not in y_by_sid:
                y_by_sid[sid] = []
            y_by_sid[sid].append(y[i].unsqueeze(0))

    result_by_sid = {}
    for sid in sorted(y_by_sid.keys()):
        sess_means, sess_stds, z_all = [], [], []
        for sess in [0, 1, 2]:
            key = (sid, sess)
            if key in z_by_sid_sess:
                z_cat = torch.cat(z_by_sid_sess[key], dim=0)
                z_all.append(z_cat)
                z_half = z_cat[: z_cat.size(0) // 2]
                mean = z_half.mean(dim=0, keepdim=True)
                std = z_half.std(dim=0, keepdim=True) + 1e-6
                sess_means.append(mean)
                sess_stds.append(std)

        if len(sess_means) == 0:
            continue

        z_all_cat = torch.cat(z_all, dim=0)
        avg_mean = torch.stack(sess_means).mean(dim=0)
        avg_std = torch.stack(sess_stds).mean(dim=0)

        z_norm = (z_all_cat - avg_mean) / avg_std
        y_cat = torch.cat(y_by_sid[sid], dim=0)

        result_by_sid[sid] = {"z_norm": z_norm, "y": y_cat}

    return result_by_sid

