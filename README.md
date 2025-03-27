# Dual-Objective Diffusion Models for EEG

This project implements a joint EEG generation and classification model using denoising diffusion techniques. It's a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## Baseline

As a starting point, we implemented and ran simple EEGNet models for each dataset to establish baseline classification accuracy. It help us evaluate the impact of diffusion-based data generation and joint training.

- **Motor Imagery**: Adapted from [EEGNet](https://github.com/amrzhd/EEGNet/tree/main) for BCI Competition IV 2a dataset.

- **SSVEP**: Adapted from [Hybrid-EEGNET-CharRNN](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor) using [Tsinghua BCI Benchmark Dataset](https://bci.med.tsinghua.edu.cn/download.html).

- **P300**: Adapted from the [P300 Speller](https://github.com/Manucar/p300-speller) repository.

- **FEIS**: Adapted from *To be updated*

> *Add a table with results.*

## Diffusion model 

> *To be updated after final experiments.*
  
## Acknowledgements

This project is based on the [DiffE repository](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG data generation. We adapted and extended core components. 
