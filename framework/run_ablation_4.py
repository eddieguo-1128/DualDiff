import os
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from config import work_dir, use_subject_wise_z_norm

# Define ablation axes
seeds = [42, 43, 44] 
ddpm_variants = ["use_ddpm",
                 "no_ddpm"] # no ddpm means no x_hat is generated
encoder_inputs = ["x", 
                  "x_hat"] # x_hat is only available when ddpm is used
decoder_inputs = ["z only"]
decoder_variants = ["use_decoder",
                     "no_decoder"] # no decoder means no decoder_out is generated 
z_norm_mode = "option2" # "option2": Z-norm in train + test; test_seen uses train stats, test_unseen uses calibration
classifier_variants = ["eegnet_classifier", 
                       "fc_classifier"] # sweep 4
classifier_inputs = ["x", "x_hat", "decoder_out", "z"] # sweep 4 // later add an option for "input_mixup"
mixup_strategy = ["none", "inputs weighted average", "inputs temporal mixup", 
                  "prior embeddings weighted average", "later embeddings weighted average"] # ablate later 
ddpm_loss = [True, False] # ablate later 
total_loss_combinations = ["alpha*{classification_loss} + beta*{reconstruction2_loss} + gamma*{contrastive_loss}", # default for sweep 3
                           "alpha*{classification_loss}",                                                    
                           "alpha*{classification_loss} + beta*{reconstruction2_loss}",                   
                           "alpha*{classification_loss} + gamma*{contrastive_loss}"] # ablate later 
classification_loss = ["CE", "MSE"] # default is CE
reconstruction2_loss = "L1" # default
contrastive_loss = "SupCon" # default, but consider adding more
alpha = 1 # default
beta = "scheduler to 0.05" # default
gamma = "scheduler to 0.2" # default

results = []

for classifier_variant in classifier_variants:
    for classifier_input in classifier_inputs:
        acc_seen_list = []
        acc_unseen_list = []
        dec_input = decoder_inputs[0]  # fixed to z only
        ddpm_variant = ddpm_variants[0]  # fixed to use_ddpm
        encoder_input = encoder_inputs[0]  # fixed to x
        decoder_variant = decoder_variants[0]  # fixed to use_decoder
        
        for seed in seeds: 
            print(f"\nRunning: classifier_variant={classifier_variant}, classifier_input={classifier_input}, seed={seed}, z_norm={z_norm_mode}")
            
            # Set environment variables
            os.environ["CLASSIFIER_VARIANT"] = classifier_variant
            os.environ["CLASSIFIER_INPUT"] = classifier_input
            os.environ["DECODER_INPUT"] = dec_input
            os.environ["SEED"] = str(seed)
            os.environ["Z_NORM_MODE"] = z_norm_mode
            os.environ["DDPM_VARIANT"] = ddpm_variant
            os.environ["ENCODER_INPUT"] = encoder_input
            os.environ["DECODER_VARIANT"] = decoder_variant

            # Construct run name
            run_name = f"{dec_input.replace(' ', '').replace('+', '_')}__classifier_variant_{classifier_variant}__classifier_input_{classifier_input}__s{seed}_z{z_norm_mode}"
            os.environ["RUN_NAME"] = run_name
            log_dir = os.path.join(work_dir, run_name, "logs")

            # Run training
            subprocess.run(["python", "framework/main.py"], check=True)

            # Load test results
            result_files = [f for f in os.listdir(log_dir) if f.startswith("test_results")]
            result_files.sort()  # take the most recent
            if not result_files:
                print(f"No result files found in {log_dir}. Skipping...")
                continue
            path = os.path.join(log_dir, result_files[-1])
            result = np.load(path, allow_pickle=True).item()

            # Compute statistics
            acc_seen = result["test1"]["accuracy"]
            acc_unseen = result["test2"]["accuracy"]

            acc_seen_list.append(acc_seen)
            acc_unseen_list.append(acc_unseen)

        # Calculate means and standard deviations after all seeds are processed
        seen_mean, seen_std = np.mean(acc_seen_list), np.std(acc_seen_list)
        unseen_mean, unseen_std = np.mean(acc_unseen_list), np.std(acc_unseen_list)

        results.append({
            "classifier_variant": classifier_variant,
            "classifier_input": classifier_input,
            "decoder_input": dec_input,
            "ddpm_variant": ddpm_variant,
            "encoder_input": encoder_input,
            "decoder_variant": decoder_variant,
            "z_norm_mode": z_norm_mode,
            "test_seen_mean": seen_mean * 100,
            "test_seen_std": seen_std * 100,
            "test_unseen_mean": unseen_mean * 100,
            "test_unseen_std": unseen_std * 100})

results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
ablation_dir = os.path.join(work_dir, "ablation_results")
os.makedirs(ablation_dir, exist_ok=True)
results_path = os.path.join(ablation_dir, f"ablation_classifier_{timestamp}.csv")
results_df.to_csv(results_path, index=False)
print(f"\nFinished. Saved results to {results_path}")

# This runs 
## classifier_variant=eegnet_classifier, classifier_input=x, seed=42, z_norm=option2
## classifier_variant=eegnet_classifier, classifier_input=x_hat, seed=42, z_norm=option2

