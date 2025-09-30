# MultiDiffNet: A Multi-Objective Diffusion Framework for Generalizable Brain Decoding

We introduce MultiDiffNet, a diffusion-based framework that generalizes to unseen subjects, supported by a new benchmark suite and evaluation protocol.

## Paper 

Accepted to the [NeurIPS 2025 Workshop on Foundation Models for the Brain and Body](https://brainbodyfm-workshop.github.io/). 

Link to the final paper will be added shortly.

## Final report

<p align="left">
  <a href="https://drive.google.com/file/d/1j9D4cUCC8CuNJWCPe7GjeGUAo6qSsD5l/view?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/View Final Report-blue?style=for-the-badge" alt="View Final Report">
  </a>
</p>

*Updated: April 30, 2025*

## Codebase

- [`baseline`](./baseline): Original EEGNet and baseline evaluation scripts.
- [`diffusion-DiffE`](./diffusion-DiffE): Original [DiffE model](https://github.com/yorgoon/DiffE) implementation.
- [`diffusion-DualDiff-Latent`](./diffusion-DualDiff-Latent): Modified DiffE `.py` files for the DualDiff-Latent model 
- [`diffusion-DualDiff-Mixup`](./diffusion-DualDiff-Mixup): Modified DiffE `.py` files for the DualDiff-Mixup model 
- [`feis-notebooks`](./feis-notebooks): Notebooks for the FEIS-specific experiments
- [`mi-notebooks`](./mi-notebooks): Notebooks for Motor Imagery-specific experiments
- [`ssvep-notebooks`](./ssvep-notebooks): Notebooks for SSVEP-specific experiments

## Acknowledgements

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## References 
- [DiffE repo](https://github.com/yorgoon/DiffE)
- [Hybrid-EEGNET-CharRNN code](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor)
- [P300 Speller code](https://github.com/Manucar/p300-speller)
- [EEGNet code](https://github.com/amrzhd/EEGNet/)    
- [FEIS code](https://github.com/scottwellington/FEIS/tree/main)
