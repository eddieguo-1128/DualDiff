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

> *TODO - Find the paper with the state-of-the-art performance on the FEIS dataset, provided it's trained on ALL subject data. Does 6.37% include ALL subjects & ALL stages?*

## Experiments 

We adapted and extended core components of the [DiffE repo](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG. 

### Long to-do:

- [x] **Prepare proper train, val, and two test sets (seen and unseen subjects)**: @all 
- [x] **Train and evaluate [DiffE repo](https://github.com/yorgoon/DiffE) on**: 
  - [x] SSVEP dataset (use either .mat or .npy data) @Ben and @Mengchun
  - [x] FEIS dataset (benchmark the performance per stage: hearing, thinking, speaking) @Parusha and Eddie
- [ ] **Experiment with local (subject-level) normalization**: @Ben and @Mengchun
  - [ ] (SSVEP) Try normalizing the input EEG
  - [ ] (SSVEP) Try normalizing the latent space learned by the diffusion (z)
  - [ ] (SSVEP) Maybe think about a learnable `subject layer` that would learn the subject space and magically adjust
- [ ] **Experiment with feature extractors**: 
  - [x] (SSVEP) Try spectrogram input instead of raw EEG signal @Ben 
- [ ] **Experiment with the model**: 
  - [x] (SSVEP) Use EEGNet as an encoder in diffusion @Mengchun
  - [x] (SSVEP) Use EEGNet as a final classifier instead of FC @Ben 
  - [x] (FEIS) Replace AvgPool with attention-based pooling @Parusha
  - [ ] Change the diffusion architecture (double block inside - more on this later)
- [ ] **Experiment with losses**:
  - [x] (SSVEP) Replace the MSE loss with the CE loss between `fc_out and y` (given it's a classification task) @Mengchun
  - [x] (SSVEP) Try a contrastive loss between `z and y` @Mengchun
  - [x] (SSVEP) Try different reconstruction losses, for example, between: (1) `x and x_hat`; (2) `x and decoder_out`; **(3)** `x_hat and decoder_out`; (4) `x, x_hat, and decoder_out` @Mengchun
  - [x] (SSVEP) Tune weighting: `loss = α * classification + β * reconstruction + γ * contrastive` @Mengchun
  - [ ] Introduce reconstruction loss in the frequency domain
- [ ] **Experiment with training procedure**:
  - [ ] Cold start: train all modules jointly from scratch
  - [ ] Step-by-step freezing: pretrain encoder/decoder, then freeze and fine-tune classifier
- [ ] **Experiment with multimodality**: @Parusha and Eddie
  - [ ] (FEIS) Think about creatively combining all three stages/modalities (hearing, thinking, speaking)
- [ ] **Run EDA & viz**: @Mengchun
  - [x] (ALL) Run t-SNE on EEG inputs VS labels (train set)
  - [x] (ALL) Run t-SNE on EEG inputs VS subjects (train set)
  - [x] (ALL) Run t-SNE on latent space (z) VS labels (train set)
  - [x] (ALL) Run t-SNE on latent space (z) VS subjects (train set)
  - [x] (ALL) Compare the diffusion output across `x`, `x_hat`, `noise`, and `decoder_out`

> *TODO - Add a summary of changes.*

## Results

> *TODO - Create an ablations sheet and add a table with results after final experiments.*
  
## Acknowledgements

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

> *TODO - To be updated.*
