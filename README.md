# Dual-Objective Diffusion Models for EEG

This project implements a joint EEG generation and classification model using denoising diffusion techniques. 

## Baseline

As a starting point, we implemented and ran simple EEGNet models for each dataset to establish baseline classification accuracy on the test set.  

| Task            | Dataset                                         | Reference implementation                                                                 | Accuracy |
|-----------------|--------------------------------------------------|-------------------------------------------------------------------------------------------|----------|
| Motor imagery   | BCI Competition IV 2a                            | [EEGNet](https://github.com/amrzhd/EEGNet/)                                                | 69.00%   |
| SSVEP           | Tsinghua BCI Benchmark Dataset                   | [Hybrid-EEGNET-CharRNN](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor) | 84.88%   |
| P300            | BCI Competition III Dataset II / P300 Speller    | [P300 Speller](https://github.com/Manucar/p300-speller)                                    | 74.00% / 77.98% |
| Brain-to-speech | Fourteen-channel EEG for Imagined Speech (FEIS)  | [FEIS](https://github.com/scottwellington/FEIS/tree/main)                                  | 6.37%    |

## Experiments 

We adapted and extended core components of the [DiffE repo](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG. 

### Long list of ideas:

- [x] **Prepare proper train, val, and two test sets (seen and unseen subjects)**: @all 
- [x] **Train and evaluate [DiffE repo](https://github.com/yorgoon/DiffE) on**: 
  - [x] SSVEP dataset
  - [x] FEIS dataset 
- [ ] **Experiment with local (subject-level) normalization**: @Ben and @Mengchun
  - [ ] Try normalizing the input EEG
  - [ ] Try normalizing the latent space learned by the diffusion (z)
- [ ] **Experiment with feature extractors**: 
  - [x] Try spectrogram input instead of raw EEG signal @Ben 
- [ ] **Experiment with the model**: 
  - [x] Use EEGNet as an encoder in diffusion @Mengchun
  - [ ] Use EEGNet as a classifier instead of FC @Ben 
  - [x] Replace AvgPool with attention-based pooling @Parusha
  - [ ] Change the architecture (double block inside - more on this later)
- [ ] **Experiment with losses**:
  - [x] Replace the MSE loss with the CE loss between `fc_out and y` (given it's a classification task) @Mengchun
  - [x] Try a contrastive loss between `z and y` @Mengchun
  - [x] Try different reconstruction losses, for example: (1) `x and x_hat`; (2) `x and decoder_out`; (3) `x_hat and decoder_out`; (4) `x, x_hat, and decoder_out` @Mengchun
  - [x] Tune weighting: `loss = α * classification + β * reconstruction + γ * contrastive` @Mengchun
  - [ ] Introduce reconstruction loss in the frequency domain
- [ ] **Experiment with training procedure**:
  - [ ] Cold start: train all modules jointly from scratch
  - [ ] Step-by-step freezing: pretrain encoder/decoder, then freeze and fine-tune classifier
- [ ] **Run EDA & viz**: @Mengchun
  - [x] Run t-SNE on EEG inputs VS labels (train set)
  - [x] Run t-SNE on EEG inputs VS subjects (train set)
  - [x] Run t-SNE on latent space (z) VS labels (train set)
  - [x] Run t-SNE on latent space (z) VS subjects (train set)
  - [x] Compare the diffusion output across `x`, `x_hat`, `noise`, and `decoder_out`

> *TODO - Add a summary of changes.*

## Results

> *TODO - Add a table with results after final experiments.*
  
## Acknowledgements

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

> *TODO - To be updated.*
