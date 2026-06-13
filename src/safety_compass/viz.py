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


def plot_phase1_outputs(csv_path: str, output_dir: str, concepts: Optional[Sequence[str]] = None):
    output = Path(output_dir)
    plot_cosine_drift(str(csv_path), str(output / "phase1_cosine_drift.png"), concepts)
    plot_auroc_degradation(str(csv_path), str(output / "phase1_auroc_degradation.png"), concepts)
