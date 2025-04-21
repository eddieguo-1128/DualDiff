# Dual-Objective Diffusion Models for EEG

This project implements a joint EEG generation and classification model using denoising diffusion techniques. 

> TODO - Update the project title at the end.

## Baseline

As a starting point, we implemented and ran simple EEGNet models for each dataset to establish baseline classification accuracy on the test set.  

| Task            | Dataset                                         | Reference implementation                                                                 | Accuracy |
|-----------------|--------------------------------------------------|-------------------------------------------------------------------------------------------|----------|
| Motor imagery   | BCI Competition IV 2a                            | [EEGNet](https://github.com/amrzhd/EEGNet/)                                                | 69.00%   |
| SSVEP           | Tsinghua BCI Benchmark Dataset                   | [Hybrid-EEGNET-CharRNN](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor) | 84.88%   |
| P300            | BCI Competition III Dataset II / P300 Speller    | [P300 Speller](https://github.com/Manucar/p300-speller)                                    | 74.00% / 77.98% |
| Brain-to-speech | Fourteen-channel EEG for Imagined Speech (FEIS)  | [FEIS](https://github.com/scottwellington/FEIS/tree/main)                                  | 6.37%    |

> *FEIS baseline is for a model trained on ALL subjects, but using only the THINKING stage.*

> *TODO - Same question to 84.88% (SSVEP)? Is it for seen, unseen, or mixed subjects?*
 
## Experiments 

We adapted and extended core components of the [DiffE repo](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG. 

### Long to-do:

- [x] **Prepare proper train, val, and two test sets (seen and unseen subjects)**: @all
- [ ] **Establish the baseline/benchmark**: 
  - [ ] (SSVEP) Check literature for SOTA (char-level) and calculate the chance/random guess @Mengchun
  - [ ] (FEIS) Check literature for SOTA (all subjects + individual vs performance per stage: hearing, thinking, speaking vs all three stages combined) and calculate the chance/random guess @Parusha and Eddie
  - [x] (SSVEP) Train and evaluate on [DiffE repo](https://github.com/yorgoon/DiffE) @Ben and @Mengchun
  - [x] (FEIS) Train and evaluate on [DiffE repo](https://github.com/yorgoon/DiffE) @Parusha and Eddie
  - [ ] (FEIS) Check the results using the **best** SSVEP model (from @Mengchun) @Parusha and Eddie
  - [ ] (FEIS) Check the results using the **best** SSVEP model (from @Ben) @Parusha and Eddie -- not ready yet
- [ ] **Experiment with normalization**: @Ben and @Mengchun
  - [x] (SSVEP) Use local (subject-level) normalization of EEG (x)
  - [x] (SSVEP) Use local (subject-level) normalization of the latent space learned by the diffusion (z) -- performs better
  - [ ] (SSVEP) Maybe think about a learnable `subject layer` that would learn the subject space and magically adjust
- [ ] **Experiment with feature extractors**: 
  - [ ] (SSVEP) Try spectrogram input instead of raw EEG signal @Ben -- some implementation issues, so maybe skip for now
- [ ] **Experiment with the model**: 
  - [x] (SSVEP) Use EEGNet as an encoder in diffusion @Mengchun
  - [ ] (SSVEP) Implement a single pipeline for creating a synthetic EEG and mixing it with a real EEG (`weight1 * x + weight2 * x_hat + weight3 * decoder_out`) for the downstream classification using EEGNet (here, `x_hat` and `decoder_out` would be a distorted/augmented version of `x`); [relevant paper](https://www.sciencedirect.com/science/article/pii/S0957417424024527) - @Ben
  - [x] (FEIS) Replace AvgPool with attention-based pooling @Parusha
  - [ ] (SSVEP) Replace AvgPool with attention-based pooling for the **best** model from @Mengchun
  - [ ] Change the diffusion architecture (double block inside - more on this later)
- [ ] **Experiment with losses**: @Mengchun
  - [x] (SSVEP) Replace the MSE loss with the CE loss between `fc_out and y` (given it's a classification task) 
  - [x] (SSVEP) Try a contrastive loss between `z and y` 
  - [x] (SSVEP) Try different reconstruction losses, for example, between: (1) `x and x_hat`; (2) `x and decoder_out`; **(3) `x_hat and decoder_out`**; (4) `x, x_hat, and decoder_out` 
  - [x] (SSVEP) Tune weighting: `loss = α * classification + β * reconstruction + γ * contrastive` 
  - [ ] Introduce reconstruction loss in the frequency domain 
- [ ] **Experiment with training procedure**:
  - [ ] Cold start: train all modules jointly from scratch
  - [ ] Step-by-step freezing: pretrain encoder/decoder, then freeze and fine-tune classifier
- [ ] **Experiment with multimodality**: @Parusha and Eddie
  - [ ] (FEIS) Think about creatively combining all three stages/modalities (hearing, thinking, speaking); maybe introducing a separate "encoder" for each stage/modality and then fusing it before passing to the diffusion ddpm block
- [x] **Run EDA & viz**: @Mengchun
  - [x] (ALL) Run t-SNE/PCA on EEG inputs VS labels (train set)
  - [x] (ALL) Run t-SNE/PCA on EEG inputs VS subjects (train set)
  - [x] (ALL) Run t-SNE/PCA on latent space (z) VS labels (train set)
  - [x] (ALL) Run t-SNE/PCA on latent space (z) VS subjects (train set)
  - [x] (ALL) Compare the diffusion output across `x`, `x_hat`, `noise`, and `decoder_out`

> *TODO - Add a summary of changes.*

## Results

> *TODO - Create an ablations sheet and add a table with results after final experiments.*
  
## Acknowledgements

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

> *TODO - To be updated.*
