import os
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from config import work_dir, use_subject_wise_z_norm

# Define ablation axes
seeds = [42]
ddpm_variants = ["use_ddpm", "no_ddpm"] # no ddpm means no x_hat is generated
encoder_inputs = ["x", "x_hat"] # x_hat is only available when ddpm is used
decoder_inputs = ["z only"]
decoder_variants = ["use_decoder", "no_decoder"] # no decoder means no decoder_out is generated 
z_local_norm_mode = "option2" # option1: directly claculate z_statistics across sessions; option2: calculate z_statistics by sessions and then average 
z_norm_mode = ["option1", "option2"] # "option2" is the default
classifier_variants = ["fc_classifier"] # "fc_classifier" is default
classifier_inputs = ["x"] # "z" is the default 
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
        if classifier_variant == "eegnet_classifier" and classifier_input == "z":
            print(f"Skipping: classifier_variant={classifier_variant}, classifier_input={classifier_input}")
            continue

        # Set decoder and DDPM variants based on classifier input requirements
        if classifier_input == "x":
            # For x input, we don't need DDPM or decoder
            ddpm_variant = "no_ddpm"
            decoder_variant = "no_decoder"
        elif classifier_input == "x_hat":
            # For x_hat input, we need DDPM but no decoder
            ddpm_variant = "use_ddpm"
            decoder_variant = "no_decoder"
        elif classifier_input == "decoder_out":
            # For decoder_out input, we need decoder but no DDPM
            ddpm_variant = "use_ddpm" 
            decoder_variant = "use_decoder"
        else:  # classifier_input == "z"
            # For z input (embedding), we use default config
            ddpm_variant = "use_ddpm"  
            decoder_variant = "use_decoder"
        
        # Fixed parameters
        dec_input = decoder_inputs[0]  # fixed to z only
        encoder_input = encoder_inputs[0]  # fixed to x
        current_z_norm = z_norm_mode[1]

        acc_seen_list = []
        acc_unseen_list = []
        
        for seed in seeds: 
            print(f"\nRunning: classifier_variant={classifier_variant}, classifier_input={classifier_input}, seed={seed}, z_norm={current_z_norm}")
            print(f"\nz_local_norm_mode={z_local_norm_mode}")

            # Set environment variables
            os.environ["CLASSIFIER_VARIANT"] = classifier_variant
            os.environ["CLASSIFIER_INPUT"] = classifier_input
            os.environ["DECODER_INPUT"] = dec_input
            os.environ["SEED"] = str(seed)
            os.environ["Z_LOCAL_NORM_MODE"] = z_local_norm_mode
            os.environ["Z_NORM_MODE"] = current_z_norm
            os.environ["DDPM_VARIANT"] = ddpm_variant
            os.environ["ENCODER_INPUT"] = encoder_input
            os.environ["DECODER_VARIANT"] = decoder_variant

            # Construct run name
            run_name = f"{dec_input.replace(' ', '').replace('+', '_')}__classifier_variant_{classifier_variant}__classifier_input_{classifier_input}__s{seed}_z{current_z_norm}"
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
            "z_local_norm_mode": z_local_norm_mode,
            "z_norm_mode": current_z_norm,
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
## classifier_variant=eegnet_classifier, classifier_input=x -> "no_decoder" + "no_ddpm" + z_norm_mode "option1"
## classifier_variant=eegnet_classifier, classifier_input=x_hat -> "no_decoder" + z_norm_mode "option1"
## classifier_variant=eegnet_classifier, classifier_input=decoder_out -> "no_ddpm" + z_norm_mode "option1"
## classifier_variant=eegnet_classifier, classifier_input=z -> skip this case + z_norm_mode "option2"
## classifier_variant=fc_classifier, classifier_input=x -> "no_decoder" + "no_ddpm" + z_norm_mode "option1"
## classifier_variant=fc_classifier, classifier_input=x_hat -> "no_decoder" + z_norm_mode "option1"
## classifier_variant=fc_classifier, classifier_input=decoder_out -> "no_ddpm" + z_norm_mode "option1"
## classifier_variant=fc_classifier, classifier_input=z -> + z_norm_mode "option2"