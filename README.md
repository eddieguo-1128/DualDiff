# Dual-Objective Diffusion Models for EEG

This project implements a joint EEG generation and classification model using denoising diffusion techniques. 

## Baseline

As a starting point, we implemented and ran simple EEGNet models for each dataset to establish baseline classification accuracy on the test set.  

| Task                          | Dataset                      | Model  | # of classes | Test accuracy                      |
|-------------------------------|-------------------------------|--------|---------|-------------------------------|
| SSVEP                         | Tsinghua BCI Benchmark        | EEGNet | 26      | 85.54% (seen) / 81.03%  (unseen)      |
| P300                          | BCI Competition III           | EEGNet | 2       | 74.00%                        |
| P300                          | P300 Speller                  | EEGNet | 2       | 77.98%                        |
| Motor Imagery                 | BCI Competition IV 2a         | EEGNet | 4       | 69.00%                        |
| Speech Production             | FEIS                          | EEGNet | 2       | 74.83%                        |
| Speech Production             | FEIS                          | EEGNet | 16      | 5.84%                         |
| Heard Speech                  | FEIS                          | EEGNet | 2       | 74.87%                        |
| Heard Speech                  | FEIS                          | EEGNet | 16      | 5.78%                         |
| Imagined Speech               | FEIS                          | EEGNet | 2       | 74.84%                        |
| Imagined Speech               | FEIS                          | EEGNet | 16      | 6.17%                         |

**Note**: 
- SSVEP baseline is tested on a mix of seen and unseen subjects
- FEIS baseline is for a model trained on ALL subjects
 
## Experiments 

We adapted and extended core components of the [DiffE repo](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG. 

### Long to-do:

- [x] **Prepare proper train, val, and two test sets (seen and unseen subjects)**: @all
- [x] **Establish the baseline/benchmark**: 
  - [x] (SSVEP) Check literature for SOTA (char-level) and calculate the chance/random guess @Mengchun
  - [x] (FEIS) Check literature for SOTA (all subjects + individual vs performance per stage: hearing, thinking, speaking vs all three stages combined) and calculate the chance/random guess @Parusha and Eddie
  - [x] (SSVEP) Train and evaluate on [DiffE repo](https://github.com/yorgoon/DiffE) @Ben and @Mengchun
  - [x] (FEIS) Train and evaluate on [DiffE repo](https://github.com/yorgoon/DiffE) @Parusha and Eddie
  - [x] (SSVEP) Get EEGNet baseline @Mengchun
  - [x] (FEIS) Get EEGNet baseline for binary vs multiclass for all three tasks @Eddie
- [ ] **Get final results on datasets of various complexity**: 
  - [x] (SSVEP) Get results for Mengchun's model @Mengchun
  - [x] (SSVEP) Get results for Ben's model @Ben 
  - [x] (FEIS) Get results for Mengchun's best model for binary vs multiclass for all three tasks @Parusha and Eddie
  - [ ] (P300 - 2 datasets) Get results for Mengchun's best model 
  - [ ] (MI) Get results for Mengchun's best model 
- [x] **Experiment with normalization**: @Ben and @Mengchun
  - [x] (SSVEP) Use local (subject-level) normalization of EEG (x)
  - [x] (SSVEP) Use local (subject-level) normalization of the latent space learned by the diffusion (z) -- performs better
  - [ ] (SSVEP) Maybe think about a learnable `subject layer` that would learn the subject space and magically adjust
- [x] **Experiment with feature extractors**: 
  - [x] (SSVEP) Try spectrogram input instead of raw EEG signal @Ben -- some implementation issues, so maybe skip for now
- [x] **Experiment with the model**: 
  - [x] (SSVEP) Use EEGNet as an encoder in diffusion @Mengchun
  - [x] (SSVEP) Implement a single pipeline for creating a synthetic EEG and mixing it with a real EEG (`weight1 * x + weight2 * x_hat + weight3 * decoder_out`) for the downstream classification using EEGNet (here, `x_hat` and `decoder_out` would be a distorted/augmented version of `x`); [relevant paper](https://www.sciencedirect.com/science/article/pii/S0957417424024527) - @Ben
  - [ ] (SSVEP) Same as above, but take Mengchun's model as stage 1 + Mixup v1 as stage 2 - @Ben
  - [ ] (SSVEP) Think of better techniques for mixups (try temporal mixups when we combine parts of inputs using some parameter that we draw from the distribution) - @Ben
  - [ ] (SSVEP) Don't hardcode alpha, beta, gamma, but introduce learnable parameters - @Ben
  - [ ] (SSVEP) Try feeding mixed input to the encoder to make latent z more robust - @Ben
  - [ ] (SSVEP) Experiment with mixups on the input vs output side (we need to perform ablations on them -> **include Ben's response from piazza**) - @Ben
  - [x] (FEIS) Replace AvgPool with attention-based pooling @Parusha
  - [x] (SSVEP) Replace AvgPool with attention-based pooling for the **best** model from @Mengchun
  - [ ] Check the decoder architecture, not sure why we are feeding x and x_hat to the decoder + other stuff
  - [ ] Try domain adversarial training
  - [ ] Change the diffusion architecture (double block inside - more on this later)
- [x] **Experiment with losses**: @Mengchun
  - [x] (SSVEP) Replace the MSE loss with the CE loss between `fc_out and y` (given it's a classification task) 
  - [x] (SSVEP) Try a contrastive loss between `z and y` 
  - [x] (SSVEP) Try different reconstruction losses, for example, between: (1) `x and x_hat`; (2) `x and decoder_out`; (3) `x_hat and decoder_out`; (4) `x, x_hat, and decoder_out` (**include this info in the report**)
  - [x] (SSVEP) Tune weighting: `loss = α * classification + β * reconstruction + γ * contrastive` 
  - [ ] Introduce reconstruction loss in the frequency domain 
- [x] **Experiment with training procedure**:
  - [x] Cold start: train all modules jointly from scratch
  - [x] Step-by-step freezing: pretrain encoder/decoder, then freeze and fine-tune classifier
- [ ] **Experiment with multimodality**: @Parusha and Eddie
  - [ ] (FEIS) Think about creatively combining all three stages/modalities (hearing, thinking, speaking); maybe introducing a separate "encoder" for each stage/modality and then fusing it before passing to the diffusion DDPM block
- [x] **Run EDA & viz**: @Mengchun
  - [x] (ALL) Run t-SNE/PCA on EEG inputs VS labels (train set)
  - [x] (ALL) Run t-SNE/PCA on EEG inputs VS subjects (train set)
  - [x] (ALL) Run t-SNE/PCA on latent space (z) VS labels (train set)
  - [x] (ALL) Run t-SNE/PCA on latent space (z) VS subjects (train set)
  - [x] (ALL) Compare the diffusion output across `x`, `x_hat`, `noise`, and `decoder_out`

## Results

> TODO - Add a table with results after final experiments

> TODO - Add the final code on the drive as .py files
> 
## Acknowledgements

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## References 
- [Hybrid-EEGNET-CharRNN code](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor)
- [P300 Speller code](https://github.com/Manucar/p300-speller)
- [EEGNet code](https://github.com/amrzhd/EEGNet/)    
- [FEIS code](https://github.com/scottwellington/FEIS/tree/main) 
