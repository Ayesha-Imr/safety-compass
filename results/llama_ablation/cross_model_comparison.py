#!/usr/bin/env python3
"""Generate cross-model comparison plots: Qwen3-8B vs Llama-3-8B."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np

from safety_compass.utils import read_csv_rows, to_float

QWEN_CSVS = {
    "Alpaca": "results/phase3/exp1/phase3_exp1_drift_log.csv",
    "Dolly": "results/phase3/exp2/phase3_exp2_drift_log.csv",
    "Code Alpaca": "results/phase3/exp3/phase3_exp3_drift_log.csv",
}

LLAMA_CSVS = {
    "Alpaca": "kaggle/llama_ablation/results_alpaca/experiment/drift_log.csv",
    "Dolly": "kaggle/llama_ablation/results_dolly/experiment/drift_log.csv",
    "Code Alpaca": "kaggle/llama_ablation/results_code_alpaca/experiment/drift_log.csv",
}

CONCEPTS = ["refusal", "sycophancy", "deception"]
CONCEPT_COLORS = {"refusal": "#1f77b4", "sycophancy": "#ff7f0e", "deception": "#2ca02c"}
OUTPUT_DIR = Path("kaggle/llama_ablation/analysis/cross_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_cosine_series(csv_path, concept):
    rows = read_csv_rows(csv_path)
    steps = [to_float(r["step"]) for r in rows]
    cosines = [to_float(r[f"{concept}_cosine_to_baseline"]) for r in rows]
    return steps, cosines


# ============================================================
# Plot 1: Side-by-side cosine drift (2x3 grid)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

for col, dataset in enumerate(["Alpaca", "Dolly", "Code Alpaca"]):
    for row, (model_name, csvs) in enumerate([("Qwen3-8B", QWEN_CSVS), ("Llama-3-8B", LLAMA_CSVS)]):
        ax = axes[row, col]
        csv_path = csvs[dataset]
        if not Path(csv_path).exists():
            csv_path = str(Path(__file__).resolve().parent.parent / csv_path)

        for concept in CONCEPTS:
            try:
                steps, cosines = load_cosine_series(csv_path, concept)
                ax.plot(steps, cosines, label=concept.capitalize(),
                        color=CONCEPT_COLORS[concept], linewidth=2)
            except Exception as e:
                print(f"  Warning: {model_name}/{dataset}/{concept}: {e}")

        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Training Step")
        if col == 0:
            ax.set_ylabel(f"{model_name}\nCosine to Baseline")
        if row == 0:
            ax.set_title(dataset, fontsize=14, fontweight="bold")
        if row == 0 and col == 2:
            ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

fig.suptitle("Concept Direction Drift: Qwen3-8B vs Llama-3-8B", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cosine_drift_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: cosine_drift_comparison.png")

# ============================================================
# Plot 2: Min cosine bar chart comparison
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for col, dataset in enumerate(["Alpaca", "Dolly", "Code Alpaca"]):
    ax = axes[col]
    x = np.arange(len(CONCEPTS))
    width = 0.35

    qwen_mins = []
    llama_mins = []

    for concept in CONCEPTS:
        for csvs, mins_list in [(QWEN_CSVS, qwen_mins), (LLAMA_CSVS, llama_mins)]:
            csv_path = csvs[dataset]
            if not Path(csv_path).exists():
                csv_path = str(Path(__file__).resolve().parent.parent / csv_path)
            rows = read_csv_rows(csv_path)
            cosines = [to_float(r[f"{concept}_cosine_to_baseline"]) for r in rows]
            mins_list.append(min(cosines))

    bars1 = ax.bar(x - width / 2, qwen_mins, width, label="Qwen3-8B", color="#4c72b0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, llama_mins, width, label="Llama-3-8B", color="#dd8452", alpha=0.85)

    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="Threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CONCEPTS])
    ax.set_title(dataset, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.1)

    for bar, val in zip(bars1, qwen_mins):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, llama_mins):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    if col == 0:
        ax.set_ylabel("Min Cosine to Baseline")
    if col == 2:
        ax.legend(loc="upper right")

fig.suptitle("Minimum Cosine Similarity: Qwen3-8B vs Llama-3-8B", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "min_cosine_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: min_cosine_comparison.png")

# ============================================================
# Plot 3: Fragility ranking heatmap
# ============================================================
fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
ax_qwen = fig.add_subplot(gs[0, 0])
ax_llama = fig.add_subplot(gs[0, 1])
cax = fig.add_subplot(gs[0, 2])

for ax, (model_name, csvs) in zip([ax_qwen, ax_llama], [("Qwen3-8B", QWEN_CSVS), ("Llama-3-8B", LLAMA_CSVS)]):
    datasets = list(csvs.keys())
    data = np.zeros((len(CONCEPTS), len(datasets)))

    for j, dataset in enumerate(datasets):
        csv_path = csvs[dataset]
        if not Path(csv_path).exists():
            csv_path = str(Path(__file__).resolve().parent.parent / csv_path)
        rows = read_csv_rows(csv_path)
        for i, concept in enumerate(CONCEPTS):
            cosines = [to_float(r[f"{concept}_cosine_to_baseline"]) for r in rows]
            data[i, j] = min(cosines)

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_yticks(range(len(CONCEPTS)))
    ax.set_yticklabels([c.capitalize() for c in CONCEPTS], fontsize=11)
    ax.set_title(model_name, fontsize=14, fontweight="bold", pad=10)

    for i in range(len(CONCEPTS)):
        for j in range(len(datasets)):
            color = "white" if data[i, j] < 0.5 else "black"
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    color=color, fontsize=12, fontweight="bold")

fig.colorbar(im, cax=cax, label="Min Cosine to Baseline")
fig.suptitle("Fragility Heatmap: Min Cosine Across Models and Datasets",
             fontsize=15, fontweight="bold", y=1.02)
plt.savefig(OUTPUT_DIR / "fragility_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fragility_heatmap.png")

print(f"\nAll cross-model plots saved to: {OUTPUT_DIR}")
