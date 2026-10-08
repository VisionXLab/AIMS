# 先对每张图片的每个 layer/head 沿 timestep 算 std，再跨 500 张图片取平均。
import os
import json
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

JSON_DIR = "/home/aiscuser/msra_workspace/code/PAI/log/chair/greedy/cosine/qwen2_5vl_start2_end_35_alpha0.18_recordw_cosine_branch_v_sgm1.5_p_sgm1.2_dw1_sgm1.1/chair_eval_500images_tokens_512_steering_weights"
OUTPUT_DIR = os.path.join(os.path.dirname(JSON_DIR), "head_weight_std_analysis")
import ipdb;ipdb.set_trace()

os.makedirs(OUTPUT_DIR, exist_ok=True)

BRANCHES = ["w_v", "w_p", "w_g"]


# ============================================================
# Accumulator
#
# branch_stats[branch][(layer, head)] = [
#     image1 temporal_std,
#     image2 temporal_std,
#     ...
# ]
# ============================================================

branch_stats = {
    branch: {}
    for branch in BRANCHES
}


json_files = sorted(
    glob.glob(os.path.join(JSON_DIR, "*.json"))
)

print(f"Found {len(json_files)} JSON files.")


# ============================================================
# Read image by image
# ============================================================

for json_path in tqdm(json_files):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data["layers"]

    for layer_str, timesteps in layers.items():

        layer_idx = int(layer_str)

        if len(timesteps) == 0:
            continue

        # ------------------------------------------------------
        # 确定 head 数量
        # ------------------------------------------------------
        num_heads = None

        for item in timesteps:
            for branch in BRANCHES:

                values = item.get(branch)

                if values is not None:
                    num_heads = len(values)
                    break

            if num_heads is not None:
                break

        if num_heads is None:
            continue

        # ------------------------------------------------------
        # 每个 head 单独统计 timestep dynamics
        # ------------------------------------------------------
        for head_idx in range(num_heads):

            for branch in BRANCHES:

                values = []

                for item in timesteps:

                    head_values = item.get(branch)

                    # 例如 timestep 0 没有 generated branch
                    if head_values is None:
                        continue

                    value = head_values[head_idx]

                    if value is None:
                        continue

                    if np.isnan(value):
                        continue

                    values.append(value)

                # timestep 太少就不计算
                if len(values) < 2:
                    continue

                temporal_std = np.std(
                    values,
                    ddof=0
                )

                key = (layer_idx, head_idx)

                if key not in branch_stats[branch]:
                    branch_stats[branch][key] = []

                branch_stats[branch][key].append(
                    temporal_std
                )


# ============================================================
# 跨 image 求平均
# ============================================================

mean_std = {}

for branch in BRANCHES:

    rows = []

    for (layer, head), std_values in branch_stats[branch].items():

        rows.append({
            "layer": layer,
            "head": head,
            "mean_temporal_std": np.mean(std_values),
            "std_across_images": np.std(std_values),
            "num_images": len(std_values),
        })

    mean_std[branch] = pd.DataFrame(rows)

    mean_std[branch].to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{branch}_head_temporal_std.csv"
        ),
        index=False
    )


# ============================================================
# Heatmap
# ============================================================

branch_titles = {
    "w_v": "Vision branch",
    "w_p": "Prefill branch",
    "w_g": "Generated branch",
}


for branch in BRANCHES:

    df = mean_std[branch]

    heatmap = df.pivot(
        index="layer",
        columns="head",
        values="mean_temporal_std"
    )

    # layer 从高到低
    heatmap = heatmap.sort_index(
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    im = ax.imshow(
        heatmap.values,
        aspect="auto"
    )

    ax.set_xticks(
        np.arange(len(heatmap.columns))
    )
    ax.set_xticklabels(
        heatmap.columns
    )

    ax.set_yticks(
        np.arange(len(heatmap.index))
    )
    ax.set_yticklabels(
        heatmap.index
    )

    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")

    ax.set_title(
        f"{branch_titles[branch]}: "
        f"Temporal Std of Steering Weights"
    )

    cbar = fig.colorbar(
        im,
        ax=ax
    )

    cbar.set_label(
        "Mean temporal std"
    )

    plt.tight_layout()

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{branch}_layer_head_temporal_std_heatmap.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")


print("\nDone.")