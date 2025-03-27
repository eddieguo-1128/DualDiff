# Dual-Objective Diffusion Models for EEG

This project implements a joint EEG generation and classification model using denoising diffusion techniques. It's a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## Baseline

| Task            | Dataset                                         | Reference Implementation                                                                 | Accuracy |
|-----------------|--------------------------------------------------|-------------------------------------------------------------------------------------------|----------|
| Motor imagery   | BCI Competition IV 2a                            | [EEGNet](https://github.com/amrzhd/EEGNet/tree/main)                                     | 69.00%   |
| SSVEP           | Tsinghua BCI Benchmark Dataset                   | [Hybrid-EEGNET-CharRNN](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor) | 84.88%   |
| P300            | BCI Competition III Dataset II / P300 Speller    | [P300 Speller](https://github.com/Manucar/p300-speller)                                  | 74.00% / 77.98% |
| Brain-to-speech | Fourteen-channel EEG for Imagined Speech (FEIS)  | *To be updated*                                                                           | 6.37%    |

## Diffusion model 

> *To be updated after final experiments.*
  
## Acknowledgements

This project is based on the [DiffE repository](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG data generation. We adapted and extended core components. 
