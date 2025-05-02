| **Combination of Losses** | **Task Paradigm** | **Model** | **Test Accuracy** |
|---------------------------|-------------------|-----------|-------------------|
| **α × CE + β × L1 + γ × SupCon**<br>α = 1<br>β = min(1.0, epoch / 100) × 0.05<br>γ = min(1.0, epoch / 50) × 0.2 (**default**) | SSVEP | DualDiff-Latent v2 (w/ attention) | **84.27% (seen)** / **87.82% (unseen)** |
| α × CE | SSVEP | DualDiff-Latent v2 | 79.72% (seen) / 84.62% (unseen) |
| α × MSE | SSVEP | DualDiff-Latent v2 | 81.82% (seen) / 86.86% (unseen) |
| α × CE + β × L1 | SSVEP | DualDiff-Latent v2 | 80.77% (seen) / 84.29% (unseen) |
| α × CE + γ × SupCon | SSVEP | DualDiff-Latent v2 | 80.07% (seen) / 83.33% (unseen) |
| α × MSE + β × L1 + γ × SupCon | SSVEP | DualDiff-Latent v2 | 82.63% (seen) / **88.14% (unseen)** <br>*(but the loss curve is weird)* |
| α × MSE2 + β × L1 + γ × SupCon | SSVEP | DualDiff-Latent v2 | 81.70% (seen) / 87.50% (unseen) <br>*(but the loss curve is weird)* |
