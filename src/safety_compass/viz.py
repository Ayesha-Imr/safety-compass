from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence


def _read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def infer_concepts(csv_path: str) -> list[str]:
    rows = _read_rows(csv_path)
    if not rows:
        return []
    suffix = "_cosine_to_baseline"
    return sorted(key[: -len(suffix)] for key in rows[0].keys() if key.endswith(suffix))


def plot_cosine_drift(
    csv_path: str,
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """Plot cosine similarity to each concept's baseline direction over time."""
    import matplotlib.pyplot as plt

    rows = _read_rows(csv_path)
    concepts = list(concepts or infer_concepts(csv_path))
    steps = [_to_float(row["step"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    for concept in concepts:
        column = f"{concept}_cosine_to_baseline"
        ax.plot(steps, [_to_float(row[column]) for row in rows], marker="o", label=concept)

    ax.axhline(0.95, linestyle="--", color="black", linewidth=1, label="GO threshold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cosine similarity to baseline")
    ax.set_title("Concept direction drift")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_auroc_degradation(
    csv_path: str,
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """Plot held-out AUROC using original fixed directions over time."""
    import matplotlib.pyplot as plt

    rows = _read_rows(csv_path)
    concepts = list(concepts or infer_concepts(csv_path))
    steps = [_to_float(row["step"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    for concept in concepts:
        column = f"{concept}_auroc_fixed"
        ax.plot(steps, [_to_float(row[column]) for row in rows], marker="o", label=concept)

    ax.axhline(0.80, linestyle="--", color="black", linewidth=1, label="Phase 0 pass line")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Fixed-direction AUROC during fine-tuning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_heatmap(
    csv_path: str,
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
):
    """Heatmap of concept metrics over training steps, per-row normalized."""
    import matplotlib.pyplot as plt
    import numpy as np

    rows = _read_rows(csv_path)
    concepts = list(concepts or infer_concepts(csv_path))
    metrics = list(metrics or ["cosine_to_baseline", "auroc_fixed", "direction_norm"])
    steps = [_to_float(row["step"]) for row in rows]

    labels = []
    data = []
    for concept in concepts:
        for metric in metrics:
            column = f"{concept}_{metric}"
            if column not in rows[0]:
                continue
            labels.append(f"{concept}\n{metric}")
            data.append([_to_float(row[column]) for row in rows])

    data = np.array(data)

    normed = np.zeros_like(data)
    for i in range(data.shape[0]):
        row_min, row_max = np.nanmin(data[i]), np.nanmax(data[i])
        if row_max - row_min > 0:
            normed[i] = (data[i] - row_min) / (row_max - row_min)
        else:
            normed[i] = 0.5

    fig, ax = plt.subplots(figsize=(max(10, len(steps) * 0.5), max(4, len(labels) * 0.6)))
    im = ax.pcolormesh(normed, cmap="RdYlGn", vmin=0, vmax=1)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                text = f"{val:.2f}" if abs(val) < 100 else f"{val:.0f}"
                ax.text(
                    j + 0.5, i + 0.5, text,
                    ha="center", va="center", fontsize=6, color="black",
                )

    ax.set_xticks([i + 0.5 for i in range(len(steps))])
    ax.set_xticklabels([f"{int(s)}" for s in steps], rotation=45, ha="right", fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(len(labels))])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Training step")
    ax.set_title("Concept metrics heatmap (per-row normalized)")
    fig.colorbar(im, ax=ax, label="Normalized value", shrink=0.8)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cross_concept_evolution(
    csv_path: str,
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """Plot cross-concept cosine similarities over training steps."""
    import matplotlib.pyplot as plt

    rows = _read_rows(csv_path)
    if not rows:
        return

    cross_cols = [k for k in rows[0].keys() if k.startswith("cross_") and k.endswith("_cosine")]
    if not cross_cols:
        return

    steps = [_to_float(row["step"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in sorted(cross_cols):
        label = col.replace("cross_", "").replace("_cosine", "").replace("_", " vs ", 1)
        ax.plot(steps, [_to_float(row[col]) for row in rows], marker="o", label=label)

    ax.axhline(0, linestyle="--", color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-concept cosine similarity")
    ax.set_title("Cross-concept direction similarity during fine-tuning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_all_outputs(csv_path: str, output_dir: str, concepts: Optional[Sequence[str]] = None):
    output = Path(output_dir)
    plot_cosine_drift(str(csv_path), str(output / "cosine_drift.png"), concepts)
    plot_auroc_degradation(str(csv_path), str(output / "auroc_degradation.png"), concepts)
    plot_metric_heatmap(str(csv_path), str(output / "metric_heatmap.png"), concepts)
    plot_cross_concept_evolution(str(csv_path), str(output / "cross_concept_evolution.png"), concepts)


def plot_phase1_outputs(csv_path: str, output_dir: str, concepts: Optional[Sequence[str]] = None):
    output = Path(output_dir)
    plot_cosine_drift(str(csv_path), str(output / "phase1_cosine_drift.png"), concepts)
    plot_auroc_degradation(str(csv_path), str(output / "phase1_auroc_degradation.png"), concepts)
