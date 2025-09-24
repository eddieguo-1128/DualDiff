import os
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
from config import work_dir

# ------------ Settings -------------
seeds = [42,43,44]  
classifier_variants = ["tidnet_classifier", "eegconformer_classifier"]  
classifier_input = "x"            # 固定：分类头吃原始 x
encoder_input = "x"               # 你的主干 encoder 仍吃 x（不影响baseline）
ddpm_variant = "no_ddpm"          # baseline不需要生成
decoder_variant = "no_decoder"    # baseline不走decoder
z_local_norm_mode = "option2"     # 无影响（baseline不做z-norm）
z_norm_mode = "option2"           # 评估时按你注释：classifier_input=x -> option1
dec_input = "z only"              # 仅用于命名，和baseline无关
# -----------------------------------

results = []

for clf in classifier_variants:
    acc_seen_list, acc_unseen_list = [], []

    for seed in seeds:
        print(f"\nRunning: classifier_variant={clf}, classifier_input={classifier_input}, seed={seed}, z_norm={z_norm_mode}")

        os.environ["CLASSIFIER_VARIANT"] = clf
        os.environ["CLASSIFIER_INPUT"] = classifier_input
        os.environ["DECODER_INPUT"] = dec_input
        os.environ["SEED"] = str(seed)
        os.environ["Z_LOCAL_NORM_MODE"] = z_local_norm_mode
        os.environ["Z_NORM_MODE"] = z_norm_mode
        os.environ["DDPM_VARIANT"] = ddpm_variant
        os.environ["ENCODER_INPUT"] = encoder_input
        os.environ["DECODER_VARIANT"] = decoder_variant

        run_name = f"{dec_input.replace(' ', '').replace('+','_')}__classifier_variant_{clf}__classifier_input_{classifier_input}__s{seed}_z{z_norm_mode}"
        os.environ["RUN_NAME"] = run_name
        log_dir = os.path.join(work_dir, run_name, "logs")

        subprocess.run(["python", "framework/main.py"], check=True)

        if not os.path.isdir(log_dir):
            print(f"[Warn] No log dir: {log_dir}")
            continue
        result_files = sorted([f for f in os.listdir(log_dir) if f.startswith("test_results")])
        if not result_files:
            print(f"[Warn] No result files in {log_dir}")
            continue
        result = np.load(os.path.join(log_dir, result_files[-1]), allow_pickle=True).item()

        acc_seen_list.append(result["test1"]["accuracy"])
        acc_unseen_list.append(result["test2"]["accuracy"])

    seen_mean, seen_std = np.mean(acc_seen_list), np.std(acc_seen_list)
    unseen_mean, unseen_std = np.mean(acc_unseen_list), np.std(acc_unseen_list)

    results.append({
        "classifier_variant": clf,
        "classifier_input": classifier_input,
        "decoder_input": dec_input,
        "ddpm_variant": ddpm_variant,
        "encoder_input": encoder_input,
        "decoder_variant": decoder_variant,
        "z_local_norm_mode": z_local_norm_mode,
        "z_norm_mode": z_norm_mode,
        "test_seen_mean": seen_mean * 100,
        "test_seen_std": seen_std * 100,
        "test_unseen_mean": unseen_mean * 100,
        "test_unseen_std": unseen_std * 100,
    })

results_df = pd.DataFrame(results)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
ablation_dir = os.path.join(work_dir, "ablation_results")
os.makedirs(ablation_dir, exist_ok=True)
out_path = os.path.join(ablation_dir, f"ablation_classifier_TWO_BASELINES_{timestamp}.csv")
results_df.to_csv(out_path, index=False)
print(f"\nFinished. Saved results to {out_path}")
