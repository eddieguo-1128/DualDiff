Ablation on decoder input

Paradigm-ssvep

Model-DualDiff-Latent v2 (w/ attention)

Test accuracy:

x + x_hat + skips (default) 84.27% (seen) / 87.82% (unseen)

x + x_hat                   83.80% (seen) / 87.18% (unseen)

x_hat + skips               80.19% (seen) / 87.82% (unseen)

x + skips                   83.10% (seen) / 86.54% (unseen)

skips                       81.70% (seen) / 84.94% (unseen)

z                           82.87% (seen) / 88.14% (unseen)

z + x                       83.80% (seen) / 87.82% (unseen)

z + x_hat                   84.27% (seen) / 88.14% (unseen) *** the best combination so far ***

z + skips                   82.05% (seen) / 86.22% (unseen)

