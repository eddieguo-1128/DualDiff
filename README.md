# Dual-Objective Diffusion Models for EEG

This project implements a joint EEG generation and classification model using denoising diffusion techniques. 

## Results 

<p align="left">
  <a href="https://drive.google.com/file/d/1j9D4cUCC8CuNJWCPe7GjeGUAo6qSsD5l/view?usp=sharing" target="_blank">
    <img src="https://img.shields.io/badge/View%20Final%20Report-blue?style=for-the-badge" alt="View Final Report">
  </a>
</p>

## Codebase

- [`baseline`](./baseline): Original EEGNet and baseline evaluation scripts.
- [`diffusion-DiffE`](./diffusion-DiffE): Original [DiffE model](https://github.com/yorgoon/DiffE) implementation.
- [`diffusion-DualDiff-Latent`](./diffusion-DualDiff-Latent): Modified DiffE `.py` files for the DualDiff-Latent model 
- [`diffusion-DualDiff-Mixup`](./diffusion-DualDiff-Mixup): Modified DiffE `.py` files for the DualDiff-Mixup model 
- [`feis-notebooks`](./feis-notebooks): Notebooks for the FEIS-specific experiments
- [`mi-notebooks`](./mi-notebooks): Notebooks for Motor Imagery-specific experiments
- [`ssvep-notebooks`](./ssvep-notebooks): Notebooks for SSVEP-specific experiments

## Paper to-do list
- [ ] Prep all code as modules to be run/reproduced quickly
- [ ] Check the variance of EEG response for the same channel across many subjects --> prove the need for a set instead of a vector format 
- [ ] Find a SOTA for creating synthetic subjects/channels (e.g., weighted average on input/embeddings/latent) and include it in Ben's experiments 
- [ ] Run a reproducibility study and report mean/std across all tasks (for P300, just pick one dataset): 
  - [ ] EEGNet
  - [ ] DualDiff-Latent v3
  - [ ] SOTA for using synthetic subjects/channels (can be done as part of Ben's experiments)
- [ ] Generate `x_hat` and `decoder_out` using **DualDiff-Latent v3**, apply different mixup strategies, and test on the **EEGNet classifier**:
  - [ ] Only `x`
  - [ ] Only `x_hat`
  - [ ] Only `decoder_out`
  - [ ] `x`, `x_hat`, `decoder_out` mixup using weighted average 
  - [ ] `x`, `x_hat`, `decoder_out` mixup using temporal mixup (+ 2-3 ablations on hyperparams)
  - [ ] Embeddings mixup (+ 2-3 ablations after which encoder layer we apply the mixup: before or after the projection layer before z)
- [ ] Rerun table 6 using **DualDiff-Latent v3**
- [ ] Run explainability study of **DualDiff-Latent v3** to understand what each part is learning 
  - [ ] Impact of decoder inputs
  - [ ] Impact of losses
- [ ] Think about changing the title; to dual-task + key insight about the latent z
      
## Experiments to-do list

We adapted and extended core components of the [DiffE repo](https://github.com/yorgoon/DiffE), which implements a denoising diffusion model for EEG. 

- [x] **Prepare proper train, val, and two test sets (seen and unseen subjects) using the same proportion across different datasets**: @all
- [x] **Establish the baseline/benchmark**: 
  - [x] (SSVEP) Check literature for SOTA (char-level) and calculate the chance/random guess @Mengchun
  - [x] (FEIS) Check literature for SOTA (all subjects + individual vs performance per stage: hearing, thinking, speaking vs all three stages combined) and calculate the chance/random guess @Parusha and Eddie
  - [x] (SSVEP) Train and evaluate on [DiffE repo](https://github.com/yorgoon/DiffE) @Ben and @Mengchun
  - [x] (FEIS) Train and evaluate on [DiffE repo](https://github.com/yorgoon/DiffE) @Parusha and Eddie
  - [x] (SSVEP) Get EEGNet baseline @Mengchun
  - [x] (FEIS) Get EEGNet baseline for binary vs multiclass for all three tasks @Eddie
- [x] **Get final results on datasets of various complexity**: 
  - [x] (SSVEP) Get results for Mengchun's model @Mengchun
  - [x] (SSVEP) Get results for Ben's model @Ben 
  - [x] (FEIS) Get results for Mengchun's best model for binary vs multiclass for all three tasks @Parusha and Eddie
  - [x] (MI) Get results for Mengchun's best model
  - [ ] (P300 - 2 datasets) Get results for Mengchun's best model 
- [x] **Experiment with normalization**: @Ben and @Mengchun
  - [x] (SSVEP) Use local (subject-level) normalization of EEG (x)
  - [x] (SSVEP) Use local (subject-level) normalization of the latent space learned by the diffusion (z) -- performs better
  - [ ] (SSVEP) Maybe think about a learnable `subject layer` that would learn the subject space and magically adjust
- [x] **Experiment with feature extractors**: 
  - [x] (SSVEP) Try spectrogram input instead of raw EEG signal @Ben -- some implementation issues, so maybe skip for now
- [x] **Experiment with the model**: 
  - [x] (SSVEP) Use EEGNet as an encoder in diffusion @Mengchun
  - [x] (SSVEP) Implement a single pipeline for creating a synthetic EEG and mixing it with a real EEG (`weight1 * x + weight2 * x_hat + weight3 * decoder_out`) for the downstream classification using EEGNet (here, `x_hat` and `decoder_out` would be a distorted/augmented version of `x`); [relevant paper](https://www.sciencedirect.com/science/article/pii/S0957417424024527) - @Ben
  - [x] (SSVEP) Same as above, but take Mengchun's model as stage 1 + Mixup v1 as stage 2 - @Ben
  - [x] (SSVEP) Think of better techniques for mixups (try temporal mixups when we combine parts of inputs using some parameter that we draw from the distribution) - @Ben
  - [x] (SSVEP) Don't hardcode alpha, beta, gamma, but introduce learnable parameters or sample from a distribution - @Ben
  - [x] (SSVEP) Try feeding mixed input to the encoder to make latent z more robust - @Ben
  - [x] (SSVEP) Experiment with mixups on the input vs output side (we need to perform ablations on them) - @Ben
  - [x] (FEIS) Replace AvgPool with attention-based pooling @Parusha
  - [x] (SSVEP) Replace AvgPool with attention-based pooling for the **best** model from @Mengchun
  - [x] Check the decoder architecture, not sure why we are feeding x and x_hat to the decoder + other stuff
  - [ ] Run DualDiff-Mixup v1 (input mixup) and DualDiff-Mixup v1 (output mixup) using DualDiff-Latent v2 as stage 1
  - [ ] Run DualDiff-Mixup v1 (input mixup) and DualDiff-Mixup v1 (output mixup) using DualDiff-Latent v2 as stage 1 + temporal input mixing algorithm instead of simple adding
  - [ ] Ablate DualDiff-Mixup v2 on different mixing parameters 
  - [ ] Run DualDiff-Mixup v2, but instead of mixing the raw inputs, try mixing the embeddings within the encoder (ablate at different layers)
  - [ ] Set up a domain adversarial training
  - [ ] Change the diffusion architecture (double block inside within z - more on this later)
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

## Acknowledgements

The project is completed as a part of [CMU 11-785: Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/S25/index.html).

## References 
- [DiffE repo](https://github.com/yorgoon/DiffE)
- [Hybrid-EEGNET-CharRNN code](https://github.com/kkipngenokoech/Hybrid-EEGNET-CharRNN-predictor)
- [P300 Speller code](https://github.com/Manucar/p300-speller)
- [EEGNet code](https://github.com/amrzhd/EEGNet/)    
- [FEIS code](https://github.com/scottwellington/FEIS/tree/main) 
