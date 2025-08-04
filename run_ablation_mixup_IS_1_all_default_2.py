import os
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from framework.config import work_dir, use_subject_wise_z_norm
#from framework.main import z_local_norm_mode

# Define ablation axes
seeds = [42, 43, 44]
#ddpm_variants = ["use_ddpm", "no_ddpm"] # no ddpm means no x_hat is generated
#encoder_inputs = ["x", "x_hat"] # x_hat is only available when ddpm is used
#decoder_inputs = ["x + x_hat + skips", "x + x_hat", "x_hat + skips", "x + skips",
#                  "skips", "z only", "z + x", "z + x_hat", "z + skips"] # "z only" is the default
#decoder_variants = ["use_decoder", "no_decoder"] # no decoder means no decoder_out is generated
z_local_norm_mode = "option2"
z_norm_mode = "option2"
classifier_variants = ["fc_classifier"] # "fc_classifier" is default
#classifier_inputs = ["x", "x_hat", "decoder_out", "z"] # "z" is the default

ddpm_reconstruction_loss = True # keep always True
classification_losses = ["CE", "MSE"] # try both, default is CE
contrastive_loss = "SupCon" # default is SupCon 
decoder_reconstruction_loss = "L1" # default is L1

alphas = [0.5, 1] # default is 1
betas = [0, "scheduler to 0.05"] # default is "scheduler to 0.05"
gammas = [0, "scheduler to 0.2"] # default is "scheduler to 0.2"

mixup_strategy = ["none", "inputs weighted average", "inputs temporal mixup", 
                  "prior embeddings weighted average", "later embeddings weighted average"] # ablate later

# Mixup params
use_mixup = True
mixing_layers = ["0", "1", "2", "3"]
warm_up_epochs = ["100", "150"]
random_ratios = [False, True]

results = []

for mixing_layer in mixing_layers:
    for warm_up_epoch in warm_up_epochs:
        for random_ratio in random_ratios:


                acc_seen_list = []
                acc_unseen_list = []

                for seed in seeds: 
                    print(f"\nRunning: mixing_layer={mixing_layer}, warm_up_epoch={warm_up_epoch}, "
                          f"random_ratio={random_ratio}, seed={seed}")
                    print(f"\nz_local_norm_mode={z_local_norm_mode}")
                            
                    # Set environment variables
                    os.environ["DDPM_RECONSTRUCTION_LOSS"] = str(ddpm_reconstruction_loss)
                    os.environ["CLASSIFICATION_LOSS"] = "CE"
                    os.environ["CONTRASTIVE_LOSS"] = contrastive_loss
                    os.environ["DECODER_RECONSTRUCTION_LOSS"] = decoder_reconstruction_loss
                    os.environ["ALPHA"] = "1"
                    os.environ["BETA"] = "scheduler to 0.05"
                    os.environ["GAMMA"] = "scheduler to 0.2"
                    os.environ["CLASSIFIER_VARIANT"] = "fc_classifier"
                    os.environ["CLASSIFIER_INPUT"] = "z"
                    os.environ["DECODER_INPUT"] =  "z only"
                    os.environ["SEED"] = str(seed)
                    os.environ["Z_LOCAL_NORM_MODE"] = z_local_norm_mode
                    os.environ["Z_NORM_MODE"] = z_norm_mode
                    os.environ["DDPM_VARIANT"] = "use_ddpm"
                    os.environ["ENCODER_INPUT"] = "x"
                    os.environ["DECODER_VARIANT"] = "use_decoder"

                    # Set mixup parameters
                    os.environ["USE_MIXUP"] = str(use_mixup)
                    os.environ["MIXING_LAYER"] = mixing_layer
                    os.environ["WARM_UP"] = warm_up_epoch
                    os.environ["RANDOM_RATIO"] = str(random_ratio)


                    # Construct run name
                    run_name = (f"mixup_ablation_mixing_layer_{mixing_layer}_warm_up_epoch_{warm_up_epoch}"
                                f"_random_ratio_{random_ratio}_s{seed}_z{z_norm_mode}")
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
                    "classification_loss": "CE",
                    "alpha": "1",
                    "beta": "scheduler to 0.05",
                    "gamma": "scheduler to 0.2",
                    "classifier_variant": os.environ["CLASSIFIER_VARIANT"],
                    "classifier_input": os.environ["CLASSIFIER_INPUT"],
                    "ddpm_variant": os.environ["DDPM_VARIANT"],
                    "encoder_input": os.environ["ENCODER_INPUT"],
                    "decoder_variant": os.environ["DECODER_VARIANT"],
                    "decoder_input": os.environ["DECODER_INPUT"],
                    "z_norm_mode": z_norm_mode,
                    "test_seen_mean": seen_mean * 100,
                    "test_seen_std": seen_std * 100,
                    "test_unseen_mean": unseen_mean * 100,
                    "test_unseen_std": unseen_std * 100,

                    "use_mixup": os.environ["USE_MIXUP"],
                    "mixing_layer": os.environ["MIXING_LAYER"],
                    "warm_up_epoch": os.environ["WARM_UP"],
                    "random_ratio": os.environ["RANDOM_RATIO"]
                })


results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
ablation_dir = os.path.join(work_dir, "ablation_results")
os.makedirs(ablation_dir, exist_ok=True)
results_path = os.path.join(ablation_dir, f"ablation_losses_{timestamp}.csv")
results_df.to_csv(results_path, index=False)
print(f"\nFinished. Saved results to {results_path}")
