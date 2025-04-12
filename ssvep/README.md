SSVEP:

01/04/2025

"ssvep_diffE_train-val.ipynb"-train:val=0.8:0.2

"ssvep_diffE_train_val_test.ipynb"-train:val:test=0.7:0.15:0.15

06/04/2025

“ssvep_diffE_train_val_test_new.ipynb”: new grouping method: split by sessions instead of subjects

"ssvep_diffE_train_val_test_new_encoder-fc.ipynb": new grouping method and only train encoder and FC

"ssvep_diffE_Spectrogram_train_val_test_new.ipynb": new grouping method and transform the 1D EEG data to 2D spectrogram

12/04/2025

"baseline_ssvep_char.ipynb": use char.npy as input and run 15-fold CV for baseline reference

"ssvep_diffE_eegnet_encoder.ipynb": replace the encoder with eegnet

"svep_diffE_eegnet_encoder_loss-ce+supcon.ipynb": modified the loss as "loss = loss_c + loss_supcon". loss_c uses cross entropy.

"svep_diffE_eegnet_encoder_loss-ce+supcon+decoder.ipynb": add loss for decoder reconstruction, add projection for improving supcon loss, and make  beta and gamma gradually increasing during training.

"ssvep_diffE_newdata.ipynb": run on data_4s from Kip's group
