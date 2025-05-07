import os
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from config import work_dir

# Define ablation axes

decoder_inputs = ["x + x_hat + skips", 
                  "x + x_hat",
                  "x_hat + skips",
                  "x + skips",
                  "skips",
                  "z only",
                  "z + x",
                  "z + x_hat",
                  "z + skips"]

seeds = [42, 43, 44]

results = []

for dec_input in decoder_inputs:
    acc_seen_list = []
    acc_unseen_list = []

    for seed in seeds:
        print(f"\nRunning: decoder_input={dec_input}, seed={seed}")

        # Set environment variables
        os.environ["DECODER_INPUT"] = dec_input
        os.environ["SEED"] = str(seed)

        # Construct run name
        run_name = f"{dec_input.replace(' ', '').replace('+', '_')}__s{seed}"
        os.environ["RUN_NAME"] = run_name
        log_dir = os.path.join(work_dir, run_name, "logs")

        # Run training
        subprocess.run(["python", "framework/main.py"], check=True)

        # Load test results
        result_files = [f for f in os.listdir(log_dir) if f.startswith("test_results")]
        result_files.sort()  # take the most recent
        path = os.path.join(log_dir, result_files[-1])
        result = np.load(path, allow_pickle=True).item()

        acc_seen = result["test1"]["accuracy"]
        acc_unseen = result["test2"]["accuracy"]

        acc_seen_list.append(acc_seen)
        acc_unseen_list.append(acc_unseen)

    # Compute statistics
    seen_mean, seen_std = np.mean(acc_seen_list), np.std(acc_seen_list)
    unseen_mean, unseen_std = np.mean(acc_unseen_list), np.std(acc_unseen_list)

    results.append({
        "decoder_input": dec_input,
        "test_seen_mean": seen_mean * 100,
        "test_seen_std": seen_std * 100,
        "test_unseen_mean": unseen_mean * 100,
        "test_unseen_std": unseen_std * 100
    })

# Save results
results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
results_df.to_csv(f"ablation_summary_{timestamp}.csv", index=False)
print(f"\nFinished. Saved results to ablation_summary_{timestamp}.csv")


#z_norm_modes = ["option1", "option2", "option3", "none"]
#encoder_inputs = ["x", "x_hat"] ## later
#ddpm_variants = ["use_ddpm", "no_ddpm"]
#loss_combinations = ... # later