import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import mne #1.9.0, need 1.24.4 version of numpy
import gdown
import scipy.io
from config import *

def zscore_norm(data):
    mean = torch.mean(data, dim=(1, 2))
    std = torch.std(data, dim=(1, 2))
    norm_data = (data - mean[:, None, None]) / std[:, None, None]
    return norm_data

def minmax_norm(data):
    min_vals = torch.min(data, dim=-1)[0]
    max_vals = torch.max(data, dim=-1)[0]
    norm_data = (data - min_vals.unsqueeze(-1)) / (
        max_vals.unsqueeze(-1) - min_vals.unsqueeze(-1))
    return norm_data

# Dataset class (currently works for SSVEP)
class EEGDataset(Dataset):
    def __init__(self, X, Y, subject_ids=None, transform=None):
        self.X = X
        self.Y = Y
        self.subject_ids = subject_ids if subject_ids is not None else torch.zeros(len(Y), dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        sid = self.subject_ids[index]
        if self.transform:
            x = self.transform(x)
        return x.squeeze(), y, sid

# Load one subject's data by session
def load_data_by_session(root_dir, subject_id, session_idx_list):
    data = np.load(os.path.join(root_dir, f"S{subject_id}_chars.npy"))
    data = data[:, session_idx_list]
    X = data.reshape(-1, 64, 250)
    Y = np.repeat(np.arange(26), len(session_idx_list))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.long)

# Split and load dataset into DataLoaders
def load_split_dataset(root_dir, num_seen, seed):
    random.seed(seed)
    all_subjects = list(range(1, 36))
    seen_subjects = random.sample(all_subjects, num_seen)
    unseen_subjects = [sid for sid in all_subjects if sid not in seen_subjects]

    split_cfg = {
        "train":  [(sid, [0, 1, 2, 3]) for sid in seen_subjects],
        "val":    [(sid, [4]) for sid in seen_subjects],
        "val2":   [(sid, [4]) for sid in seen_subjects],
        "test1":  [(sid, [5]) for sid in seen_subjects],
        "test2":  [(sid, [0, 1, 2, 3, 4, 5]) for sid in unseen_subjects]}

    print(f"[Split] Seen subjects (train/val/test1): {seen_subjects}")
    print(f"[Split] Unseen subjects (test2): {unseen_subjects}")

    loaders = {}
    subject_id_dict = {}

    for split, sid_sess in split_cfg.items():
        X_all, Y_all, subject_ids = [], [], []
        for sid, sess in sid_sess:
            X, Y = load_data_by_session(root_dir, sid, sess)
            X_all.append(X)
            Y_all.append(Y)
            subject_ids.extend([sid] * len(Y))

        X_all = torch.cat(X_all, dim=0)
        Y_all = torch.cat(Y_all, dim=0)
        dataset = EEGDataset(X_all, Y_all, subject_ids=torch.tensor(subject_ids, dtype=torch.long))
        loaders[split] = DataLoader(dataset, batch_size=32, shuffle=(split in ["train", "val2"]))
        subject_id_dict[split] = torch.tensor(subject_ids, dtype=torch.long)

    for split in ["train", "val"]:
        if split in subject_id_dict:
            loaders[f"{split}_subjects"] = subject_id_dict[split]

    return loaders

#MI
def split_by_class_and_run(Y, seed=44, trials_per_run=12, n_classes=4, n_runs=6, train_count=8, val_count=2, test_count=2, num_sessions=2):
    random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in range(n_classes):
        cls_indices = (Y == cls).nonzero(as_tuple=True)[0]
        expected_count = trials_per_run * n_runs * num_sessions
        assert len(cls_indices) == expected_count, \
            f"Class {cls} expected {expected_count}, got {len(cls_indices)}"

        for run in range(n_runs):
            run_trials = cls_indices[run * trials_per_run: (run + 1) * trials_per_run].tolist()
            random.shuffle(run_trials)
            train_idx.extend(run_trials[:train_count])
            val_idx.extend(run_trials[train_count:train_count + val_count])
            test_idx.extend(run_trials[train_count + val_count:train_count + val_count + test_count])
    return train_idx, val_idx, test_idx

def MI_load_data_by_session(root_dir, subject_id, session_folders, label_dir):
    """
    root_dir/
      first_session/
        A01T_cleaned.fif … A09T_cleaned.fif
      second_session/
        A01E_cleaned.fif … A09E_cleaned.fif

    session_folders: list of folder names, e.g. ["first_session"] or ["second_session"]
    """
    X_list, Y_list = [], []

    for folder in session_folders:
        prefix = "T" if folder == "first_session" else "E"
        fname = f"A{subject_id:02d}{prefix}.fif"
        fpath = os.path.join(root_dir, folder, fname)
        raw = mne.io.read_raw_fif(fpath, preload=True, verbose=False)

        if folder == "first_session":
            # MI cue as '769'~'772'，mapping as 0–3 labels
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            motor_keys = ['769', '770', '771', '772']
            motor_event_id = {k: v for k, v in event_id.items() if k in motor_keys}
            if len(motor_event_id) < 4:
                raise ValueError(f"{fname} missing MI cues. Found: {event_id}")
            events = np.array([e for e in events if e[2] in motor_event_id.values()])
            label_map = {
                motor_event_id['769']: 0,
                motor_event_id['770']: 1,
                motor_event_id['771']: 2,
                motor_event_id['772']: 3,
            }
            labels = np.array([label_map[e[-1]] for e in events])

        elif folder == "second_session":
            assert label_dir is not None, "Must provide label_dir for second_session"
            mat_path = os.path.join(label_dir, f"A{subject_id:02d}E.mat")
            mat = scipy.io.loadmat(mat_path)
            label_key = [k for k in mat if "label" in k.lower() and not k.startswith("__")][0]
            labels = mat[label_key].squeeze()
            assert labels.shape[0] == 288, f"{mat_path} expected 288 labels, got {labels.shape}"
            labels = labels - 1  # Convert from 1–4 to 0–3

            # cue '783' as unknown MI tasks
            mask = raw.annotations.description == '783'
            assert mask.sum() == 288, f"Expected 288 '783' cue annotations, got {mask.sum()}"

            cue_annot = raw.annotations[mask]
            raw.set_annotations(cue_annot)
            events, _ = mne.events_from_annotations(raw, verbose=False)

        epochs = mne.Epochs(
            raw, events,
            tmin=0.0,
            tmax=4.0,
            baseline=None,
            preload=True,
            verbose=False,
            event_repeated="drop"
        )
        data = epochs.get_data()
        X_list.append(torch.from_numpy(data).float())
        Y_list.append(torch.from_numpy(labels).long())

    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    return X, Y

def MI_load_split_dataset(root_dir, num_seen, seed=43):
    random.seed(seed)

    all_subjects   = list(range(1, 10))
    seen_subjects  = random.sample(all_subjects, num_seen)
    unseen_subjects = [sid for sid in all_subjects if sid not in seen_subjects]

    split_cfg = {
        "train_val_test1": [(sid, ["first_session", "second_session"]) for sid in seen_subjects],
        "test2":     [(sid, ["first_session", "second_session"]) for sid in unseen_subjects],
    }

    print(f"[Split] Seen (train/val/test1): {seen_subjects}")
    print(f"[Split] Unseen (test2):        {unseen_subjects}")

    loaders = {}
    subject_id_dict = {}

    # load train_val data first
    X_train_all, Y_train_all, train_subject_ids = [], [], []
    X_val_all, Y_val_all, val_subject_ids = [], [], []
    X_test1_all, Y_test1_all, test1_subject_ids = [], [], []

    for sid in seen_subjects:
        for session in ["first_session", "second_session"]:
            X, Y = MI_load_data_by_session(root_dir, sid, [session], label_dir)
            
            train_idx, val_idx, test_idx = split_by_class_and_run(
                Y, seed=seed, num_sessions=1
            )

            X_train_all.append(X[train_idx])
            Y_train_all.append(Y[train_idx])
            train_subject_ids.extend([sid] * len(train_idx))

            X_val_all.append(X[val_idx])
            Y_val_all.append(Y[val_idx])
            val_subject_ids.extend([sid] * len(val_idx))

            X_test1_all.append(X[test_idx])
            Y_test1_all.append(Y[test_idx])
            test1_subject_ids.extend([sid] * len(test_idx))

    loaders["train"] = DataLoader(
        EEGDataset(torch.cat(X_train_all), torch.cat(Y_train_all),
                   subject_ids=torch.tensor(train_subject_ids, dtype=torch.long)),
        batch_size=32, shuffle=True)
    subject_id_dict["train"] = torch.tensor(train_subject_ids, dtype=torch.long)

    loaders["val"] = DataLoader(
        EEGDataset(torch.cat(X_val_all), torch.cat(Y_val_all),
                   subject_ids=torch.tensor(val_subject_ids, dtype=torch.long)),
        batch_size=32, shuffle=False)
    subject_id_dict["val"] = torch.tensor(val_subject_ids, dtype=torch.long)

    loaders["test1"] = DataLoader(
        EEGDataset(torch.cat(X_test1_all), torch.cat(Y_test1_all),
                   subject_ids=torch.tensor(test1_subject_ids, dtype=torch.long)),
        batch_size=32, shuffle=False)
    subject_id_dict["test1"] = torch.tensor(test1_subject_ids, dtype=torch.long)

    # Process unseen subjects: use all trials
    X_all, Y_all, subject_ids = [], [], []
    for sid in unseen_subjects:
        X, Y = MI_load_data_by_session(root_dir, sid, ["first_session", "second_session"],label_dir)
        X_all.append(X)
        Y_all.append(Y)
        subject_ids.extend([sid] * len(Y))

    loaders["test2"] = DataLoader(
        EEGDataset(torch.cat(X_all), torch.cat(Y_all),
                   subject_ids=torch.tensor(subject_ids, dtype=torch.long)),
        batch_size=32, shuffle=False)
    subject_id_dict["test2"] = torch.tensor(subject_ids, dtype=torch.long)

    for split in ["train", "val", "test1", "test2"]:
        subj_tensor = subject_id_dict[split]
        total_trials = subj_tensor.shape[0]
        unique_subjects = subj_tensor.unique()
        #print(f"\n[Check] Split: {split}")
        #print(f"  Total trials: {total_trials}")
        #print(f"  Unique subjects: {list(unique_subjects.numpy())}")
        for sid in unique_subjects:
            sid_count = (subj_tensor == sid).sum().item()
            #print(f"    Subject {sid.item()}: {sid_count} trials")

    return loaders