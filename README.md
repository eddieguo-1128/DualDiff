# MultiDiffNet: A Multi-Objective Diffusion Framework for Generalizable Brain Decoding

We introduce MultiDiffNet, a diffusion-based framework that generalizes to unseen subjects, supported by a new benchmark suite and evaluation protocol.

## Paper 

Accepted to the [NeurIPS 2025 Workshop on Foundation Models for the Brain and Body](https://brainbodyfm-workshop.github.io/). 

<p align="left">
  <a href="https://arxiv.org/abs/2511.18294" target="_blank">
    <img src="https://img.shields.io/badge/View%20ArXiv%20Paper-blue?style=for-the-badge" alt="ArXiv Paper">
  </a>
</p>

## Codebase

- [`framework`](./framework): Core framework with main training scripts, models, and configuration
- [`ablations`](./ablations): Ablation study scripts for running multiple experiments with different parameter combinations
- [`notebooks`](./notebooks): Jupyter notebooks organized by dataset (baseline, FEIS, Motor Imagery, SSVEP experiments)
- [`archive`](./archive): Previous implementations including DiffE and DualDiff-Latent models

## How to run

The configuration is controlled by parameters in `framework/config.py`. Modify this file to change:
- Work directory (todo: pick "local" or "drive")
- Dataset directory (todo: specify the location)
- Other parameters

### Running a single experiment

To run a single experiment with the current configuration:

```bash
python framework/main.py
```

### Running ablation studies

To run multiple experiments with different parameter combinations, use the ablation scripts:

```bash
python ablations/run_ablation_0.py    # Basic parameter sweep
python ablations/run_ablation_1.py    # Different configurations
# ... other ablation scripts
```

Each ablation script automatically runs multiple experiments with different parameter settings and saves results to CSV files in the work directory.

### Using the runner notebook

Alternatively, you can use the provided runner notebook for interactive execution:

```bash
jupyter notebook framework/runner.ipynb
```

This notebook provides a step-by-step interface for configuring and running the training process.

## Acknowledgements

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## References 
- [DiffE repo](https://github.com/yorgoon/DiffE)
- [Hybrid-EEGNET-CharRNN code](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor)
- [P300 Speller code](https://github.com/Manucar/p300-speller)
- [EEGNet code](https://github.com/amrzhd/EEGNet/)    
- [FEIS code](https://github.com/scottwellington/FEIS/tree/main)
