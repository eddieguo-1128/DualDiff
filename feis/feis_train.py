import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score, top_k_accuracy_score
)
from tqdm import tqdm
from ema_pytorch import EMA
from models import ConditionalUNet, DDPM, Encoder, Decoder, LinearClassifier, DiffE
from utils import zscore_norm, minmax_norm, EEGDataset

# Configuration
class Config:
    # Data parameters
    DATA_ROOT = "/feis/FEIS/data_eeg"
    SUBJECT_IDS = [str(i).zfill(2) for i in range(1, 22)]
    TASK_TYPE = 'thinking'
    WINDOW_LEN = 256
    WINDOW_STEP = 128
    TARGET_CHANNELS = 16
    TEST_SIZE = 0.2
    
    # Training parameters
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_TEST = 260
    SEED = 42
    NUM_EPOCHS = 500
    TEST_PERIOD = 1
    
    # Model parameters
    NUM_CLASSES = 16
    DDPM_DIM = 128
    ENCODER_DIM = 256
    FC_DIM = 512
    N_T = 1000
    
    # Optimization
    BASE_LR = 9e-5
    MAX_LR = 1.5e-3
    ALPHA = 0.1
    EMA_BETA = 0.95
    SCHEDULER_STEP = 150
    SCHEDULER_GAMMA = 0.9998

# EEG Data Processor
class EEGDataProcessor:
    NON_EEG_COLS = ['Time:256Hz', 'Epoch', 'Label', 'Stage', 'Flag']
    
    @classmethod
    def load_single_subject(cls, data_root, subject_id, task_type, selected_channels=None):
        """Load and process EEG data for a single subject"""
        csv_path = os.path.join(data_root, subject_id, task_type, f"{task_type}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")

        eeg_df = pd.read_csv(csv_path)
        channels = selected_channels or [c for c in eeg_df.columns if c not in cls.NON_EEG_COLS]
        
        return eeg_df[channels].values, cls._extract_labels(eeg_df)

    @staticmethod
    def _extract_labels(eeg_df):
        """Extract and encode labels from EEG dataframe"""
        if 'Label' not in eeg_df.columns:
            return np.zeros(len(eeg_df))
        
        labels = eeg_df['Label'].values
        label_map = {l: i for i, l in enumerate(np.unique(labels))}
        return np.vectorize(label_map.get)(labels)

    @classmethod
    def create_windows(cls, eeg_data, labels, window_len, window_step):
        """Create time windows from EEG data"""
        features, targets = [], []
        
        for i in range(0, len(eeg_data) - window_len + 1, window_step):
            features.append(eeg_data[i:i+window_len, :].T)
            targets.append(labels[i + window_len // 2])
            
        return (
            zscore_norm(torch.tensor(np.array(features), dtype=torch.float32)),
            torch.tensor(np.array(targets), dtype=torch.long)
        )

# Data Loader Factory
class DataLoaderFactory:
    @staticmethod
    def create_loaders(config):
        """Create train and test data loaders"""
        features, targets = DataLoaderFactory._load_all_subjects(config)
        features = DataLoaderFactory._pad_channels(features, config.TARGET_CHANNELS)
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets,
            test_size=config.TEST_SIZE,
            shuffle=True,
            stratify=targets,
            random_state=config.SEED
        )
        
        return (
            DataLoader(EEGDataset(X_train, y_train), 
                     batch_size=config.BATCH_SIZE_TRAIN, shuffle=True),
            DataLoader(EEGDataset(X_test, y_test), 
                     batch_size=config.BATCH_SIZE_TEST, shuffle=False)
        )

    @staticmethod
    def _load_all_subjects(config):
        """Load and combine data from all subjects"""
        all_features, all_targets = [], []
        
        for subject_id in config.SUBJECT_IDS:
            try:
                eeg_data, labels = EEGDataProcessor.load_single_subject(
                    config.DATA_ROOT, subject_id, config.TASK_TYPE
                )
                features, targets = EEGDataProcessor.create_windows(
                    eeg_data, labels, config.WINDOW_LEN, config.WINDOW_STEP
                )
                all_features.append(features)
                all_targets.append(targets)
                print(f"Loaded subject {subject_id}")
            except Exception as e:
                print(f"Skipping subject {subject_id}: {str(e)}")
        
        if not all_features:
            raise RuntimeError("No data loaded - check configuration")
        
        return torch.cat(all_features), torch.cat(all_targets)

    @staticmethod
    def _pad_channels(features, target_channels):
        """Pad EEG channels to match target size"""
        if features.shape[1] >= target_channels:
            return features
            
        padding = torch.zeros((
            features.size(0),
            target_channels - features.size(1),
            features.size(2)
        ), dtype=features.dtype)
        
        return torch.cat([features, padding], dim=1)

# Model Trainer
class EEGModelTrainer:
    def __init__(self, config, train_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        sample_features, _ = next(iter(train_loader))
        self.models = self._init_models(sample_features.shape[1])
        self.optimizers = self._init_optimizers()
        self.schedulers = self._init_schedulers()
        self.best_metrics = {k: 0 for k in ['accuracy', 'f1', 'recall', 'precision', 'auc']}

    def _init_models(self, in_channels):
        """Initialize all model components"""
        ddpm_model = ConditionalUNet(in_channels, n_feat=self.config.DDPM_DIM).to(self.device)
        ddpm = DDPM(
            nn_model=ddpm_model,
            betas=(1e-6, 1e-2),
            n_T=self.config.N_T,
            device=self.device
        ).to(self.device)
        
        encoder = Encoder(in_channels, dim=self.config.ENCODER_DIM).to(self.device)
        decoder = Decoder(in_channels, n_feat=self.config.DDPM_DIM, 
                         encoder_dim=self.config.ENCODER_DIM).to(self.device)
        fc = LinearClassifier(self.config.ENCODER_DIM, self.config.FC_DIM, 
                            emb_dim=self.config.NUM_CLASSES).to(self.device)
        
        return {
            'ddpm': ddpm,
            'diffe': DiffE(encoder, decoder, fc).to(self.device),
            'fc_ema': EMA(fc, beta=self.config.EMA_BETA, 
                         update_after_step=100, update_every=10)
        }

    def _init_optimizers(self):
        """Initialize optimizers"""
        return {
            'ddpm': optim.RMSprop(self.models['ddpm'].parameters(), lr=self.config.BASE_LR),
            'diffe': optim.RMSprop(self.models['diffe'].parameters(), lr=self.config.BASE_LR)
        }

    def _init_schedulers(self):
        """Initialize learning rate schedulers"""
        return {
            'ddpm': self._create_scheduler(self.optimizers['ddpm']),
            'diffe': self._create_scheduler(self.optimizers['diffe'])
        }

    def _create_scheduler(self, optimizer):
        return optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.config.BASE_LR,
            max_lr=self.config.MAX_LR,
            step_size_up=self.config.SCHEDULER_STEP,
            mode="exp_range",
            cycle_momentum=False,
            gamma=self.config.SCHEDULER_GAMMA
        )

    def train(self):
        """Main training loop"""
        progress_bar = tqdm(total=self.config.NUM_EPOCHS, desc="Training")
        
        for epoch in range(self.config.NUM_EPOCHS):
            self._train_epoch(epoch)
            
            if epoch >= self.config.TEST_PERIOD and epoch % self.config.TEST_PERIOD == 0:
                self._validate(epoch, progress_bar)
            
            progress_bar.update(1)
        
        progress_bar.close()
        return self.best_metrics

    def _train_epoch(self, epoch):
        """Single training epoch"""
        self.models['ddpm'].train()
        self.models['diffe'].train()
        
        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.long().to(self.device)
            
            # DDPM forward pass
            self.optimizers['ddpm'].zero_grad()
            x_hat, down, up, noise, t = self.models['ddpm'](features)
            loss_ddpm = F.l1_loss(x_hat, features, reduction='none').mean()
            loss_ddpm.backward()
            self.optimizers['ddpm'].step()
            
            # Diff-E forward pass
            self.optimizers['diffe'].zero_grad()
            decoder_out, fc_out = self.models['diffe'](features, (x_hat, down, up, t))
            
            y_cat = F.one_hot(targets, num_classes=self.config.NUM_CLASSES).float()
            loss_gap = F.l1_loss(decoder_out, loss_ddpm.detach())
            loss_class = F.mse_loss(fc_out, y_cat)
            loss_total = loss_gap + self.config.ALPHA * loss_class
            loss_total.backward()
            
            self.optimizers['diffe'].step()
            self.models['fc_ema'].update()
            
            # Update schedulers
            self.schedulers['ddpm'].step()
            self.schedulers['diffe'].step()

    def _validate(self, epoch, progress_bar):
        """Validation and metric tracking"""
        metrics = self._evaluate()
        self._update_best_metrics(metrics)
        
        progress_bar.set_description(
            f"Best Acc: {self.best_metrics['accuracy']*100:.2f}% | "
            f"Current Acc: {metrics['accuracy']*100:.2f}%"
        )

    def _evaluate(self):
        """Evaluate model on test set"""
        self.models['fc_ema'].eval()
        all_targets, all_preds = [], []
        
        with torch.no_grad():
            for features, targets in self.test_loader:
                features = features.to(self.device)
                encoder_out = self.models['diffe'].encoder(features)
                y_hat = F.softmax(self.models['fc_ema'](encoder_out[1]), dim=1)
                
                all_targets.append(targets.cpu())
                all_preds.append(y_hat.cpu())
        
        targets = torch.cat(all_targets).numpy()
        preds = torch.cat(all_preds).numpy()
        labels = np.arange(self.config.NUM_CLASSES)
        
        return {
            'accuracy': top_k_accuracy_score(targets, preds, k=1, labels=labels),
            'f1': f1_score(targets, preds.argmax(1), average="macro", labels=labels),
            'recall': recall_score(targets, preds.argmax(1), average="macro", labels=labels),
            'precision': precision_score(targets, preds.argmax(1), average="macro", labels=labels),
            'auc': roc_auc_score(targets, preds, average="macro", multi_class="ovo", labels=labels)
        }

    def _update_best_metrics(self, metrics):
        """Update best metrics tracking"""
        for metric in self.best_metrics:
            if metrics[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = metrics[metric]

# Main execution
def main():
    # Set random seeds
    random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    
    # Prepare data
    train_loader, test_loader = DataLoaderFactory.create_loaders(Config)
    
    # Initialize and train model
    trainer = EEGModelTrainer(Config, train_loader, test_loader)
    best_metrics = trainer.train()
    
    # Print results
    print("\nTraining completed. Best metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()