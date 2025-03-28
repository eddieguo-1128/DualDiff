# Dual-Objective Diffusion Models for EEG

This project implements a joint EEG generation and classification model using denoising diffusion techniques. It's a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## Baseline

As a starting point, we implemented and ran simple EEGNet models for each dataset to establish baseline classification accuracy on the test set.  

| Task            | Dataset                                         | Reference implementation                                                                 | Accuracy |
|-----------------|--------------------------------------------------|-------------------------------------------------------------------------------------------|----------|
| Motor imagery   | BCI Competition IV 2a                            | [EEGNet](https://github.com/amrzhd/EEGNet/)                                             | 69.00%   |
| SSVEP           | Tsinghua BCI Benchmark Dataset                   | [Hybrid-EEGNET-CharRNN](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor) | 84.88%   |
| P300            | BCI Competition III Dataset II / P300 Speller    | [P300 Speller](https://github.com/Manucar/p300-speller)                                  | 74.00% / 77.98% |
| Brain-to-speech | Fourteen-channel EEG for Imagined Speech (FEIS)  | [FEIS](https://github.com/scottwellington/FEIS/tree/main)                  | 6.37%    |

## Experiments 

We adapted and extended core components of the [DiffE repo](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG. 

### Long list of ideas:

- [ ] Train and evaluate [DiffE repo](https://github.com/yorgoon/DiffE) on: 
  - [ ] SSVEP
  - [ ] FEIS
- [ ] **Experiment with feature extractors**: 
  - [ ] Try spectrogram input instead of raw signal
- [ ] **Experiment with the model**: 
  - [ ] Check skip connections in encoder/decoder
  - [ ] Replace AvgPool with attention-based pooling before the classifier
  - [ ] Change the architecture (double block inside - more on this later)
- [ ] **Experiment with losses**:
  - [ ] Introduce reconstruction loss in the frequency domain
  - [ ] Explore contrastive loss between real vs. generated EEG in latent space
  - [ ] Tune loss weighting: `α * classification + β * diffusion`
  - [ ] Implement classifier-free guidance style objective (conditioned vs. unconditioned paths)
- [ ] **Experiment with training procedure**:
  - [ ] Cold start: train all modules jointly from scratch
  - [ ] Step-by-step freezing: pretrain encoder/decoder, then freeze and fine-tune classifier
- [ ] **Run EDA & viz**:
  - [ ] Generate samples of synthetic EEG and compare visually/statistically with real EEG  

> *TODO - Add a summary of changes.*

## Results

> *TODO - Add a table with results after final experiments.*
  
## Acknowledgements

> *TODO - To be updated.*
