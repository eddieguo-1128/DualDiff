import os
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from config import work_dir, use_subject_wise_z_norm, task
import warnings
warnings.filterwarnings("ignore", category=UserWarning, 
                       module="sklearn.metrics._classification")
warnings.filterwarnings("ignore", message="This filename .* does not conform to MNE naming conventions.*",
                        category=RuntimeWarning, module="mne.io")

# Define ablation axes
seeds = [42, 43, 44]
ddpm_variants = ["use_ddpm", "no_ddpm"] # no ddpm means no x_hat is generated
encoder_inputs = ["x", "x_hat"] # x_hat is only available when ddpm is used
decoder_inputs = ["x + x_hat + skips", "x + x_hat", "x_hat + skips", "x + skips",
                  "skips", "z only", "z + x", "z + x_hat", "z + skips"] # "z only" is the default
decoder_variants = ["use_decoder", "no_decoder"] # no decoder means no decoder_out is generated 
z_local_norm_mode = "option2" # option1: directly claculate z_statistics across sessions; option2: calculate z_statistics by sessions and then average 
z_norm_mode = "option2" 
classifier_variants = ["eegnet_classifier", "fc_classifier"] # "fc_classifier" is default
classifier_inputs = ["x", "x_hat", "decoder_out", "z"] # "z" is the default 

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

for ddpm_variant in ddpm_variants:
    for encoder_input in encoder_inputs:
        if ddpm_variant == "no_ddpm" and encoder_input == "x_hat":
            print("Skipping incompatible combo: no_ddpm + x_hat")
            continue
        for decoder_variant in decoder_variants:

            acc_seen_list = []
            acc_unseen_list = []
            
            for seed in seeds: 
                print(f"\nRunning: ddpm_variant={ddpm_variant}, encoder_input={encoder_input}, decoder_variant={decoder_variant}, seed={seed}, z_norm={z_norm_mode}")
                print(f"\nz_local_norm_mode={z_local_norm_mode}")

                # Set environment variables
                os.environ["CLASSIFIER_VARIANT"] = "fc_classifier"  
                os.environ["CLASSIFIER_INPUT"] = "z"
                os.environ["DECODER_INPUT"] = "z only"
                os.environ["SEED"] = str(seed)
                os.environ["Z_LOCAL_NORM_MODE"] = z_local_norm_mode
                os.environ["Z_NORM_MODE"] = z_norm_mode
                os.environ["DDPM_VARIANT"] = ddpm_variant
                os.environ["ENCODER_INPUT"] = encoder_input
                os.environ["DECODER_VARIANT"] = decoder_variant

                # Construct run name
                run_name = f"task_{task}__ddpm_{ddpm_variant}__encoder_{encoder_input}__decoder_{decoder_variant}__s{seed}_z{z_norm_mode}"
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
                "classifier_variant": os.environ["CLASSIFIER_VARIANT"],
                "classifier_input": os.environ["CLASSIFIER_INPUT"],
                "decoder_input": os.environ["DECODER_INPUT"],
                "ddpm_variant": ddpm_variant,
                "encoder_input": encoder_input,
                "decoder_variant": decoder_variant,
                "z_local_norm_mode": z_local_norm_mode,
                "z_norm_mode": z_norm_mode,
                "test_seen_mean": seen_mean * 100,
                "test_seen_std": seen_std * 100,
                "test_unseen_mean": unseen_mean * 100,
                "test_unseen_std": unseen_std * 100})

results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
ablation_dir = os.path.join(work_dir, "ablation_results")
os.makedirs(ablation_dir, exist_ok=True)
results_path = os.path.join(ablation_dir, f"task_{task}_ablation_ddpm_encoder_decoder_{timestamp}.csv")
results_df.to_csv(results_path, index=False)
print(f"\nFinished. Saved results to {results_path}")
