from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from safety_compass.utils import (
    DEFAULT_PLOT_DPI,
    DRIFT_THRESHOLD,
    MIN_AUROC_DEFAULT,
    read_csv_rows,
    to_float,
)


def infer_concepts(csv_path: str) -> list[str]:
    rows = read_csv_rows(csv_path)
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

    rows = read_csv_rows(csv_path)
    concepts = list(concepts or infer_concepts(csv_path))
    steps = [to_float(row["step"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    for concept in concepts:
        column = f"{concept}_cosine_to_baseline"
        ax.plot(steps, [to_float(row[column]) for row in rows], marker="o", label=concept)

    ax.axhline(DRIFT_THRESHOLD, linestyle="--", color="black", linewidth=1, label="GO threshold")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cosine similarity to baseline")
    ax.set_title("Concept direction drift")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_auroc_degradation(
    csv_path: str,
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """Plot held-out AUROC using original fixed directions over time."""
    import matplotlib.pyplot as plt

    rows = read_csv_rows(csv_path)
    concepts = list(concepts or infer_concepts(csv_path))
    steps = [to_float(row["step"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    for concept in concepts:
        column = f"{concept}_auroc_fixed"
        ax.plot(steps, [to_float(row[column]) for row in rows], marker="o", label=concept)

    ax.axhline(MIN_AUROC_DEFAULT, linestyle="--", color="black", linewidth=1, label="AUROC pass line")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Fixed-direction AUROC during fine-tuning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
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

    rows = read_csv_rows(csv_path)
    concepts = list(concepts or infer_concepts(csv_path))
    metrics = list(metrics or ["cosine_to_baseline", "auroc_fixed", "direction_norm"])
    steps = [to_float(row["step"]) for row in rows]

    labels = []
    data = []
    for concept in concepts:
        for metric in metrics:
            column = f"{concept}_{metric}"
            if column not in rows[0]:
                continue
            labels.append(f"{concept}\n{metric}")
            data.append([to_float(row[column]) for row in rows])

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
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_cross_concept_evolution(
    csv_path: str,
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """Plot cross-concept cosine similarities over training steps."""
    import matplotlib.pyplot as plt

    rows = read_csv_rows(csv_path)
    if not rows:
        return

    cross_cols = [k for k in rows[0].keys() if k.startswith("cross_") and k.endswith("_cosine")]
    if not cross_cols:
        return

    steps = [to_float(row["step"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in sorted(cross_cols):
        label = col.replace("cross_", "").replace("_cosine", "").replace("_", " vs ", 1)
        ax.plot(steps, [to_float(row[col]) for row in rows], marker="o", label=label)

    ax.axhline(0, linestyle="--", color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-concept cosine similarity")
    ax.set_title("Cross-concept direction similarity during fine-tuning")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_all_outputs(csv_path: str, output_dir: str, concepts: Optional[Sequence[str]] = None):
    output = Path(output_dir)
    plot_cosine_drift(str(csv_path), str(output / "cosine_drift.png"), concepts)
    plot_auroc_degradation(str(csv_path), str(output / "auroc_degradation.png"), concepts)
    plot_metric_heatmap(str(csv_path), str(output / "metric_heatmap.png"), concepts)
    plot_cross_concept_evolution(str(csv_path), str(output / "cross_concept_evolution.png"), concepts)


def plot_single_experiment(csv_path: str, output_dir: str, concepts: Optional[Sequence[str]] = None):
    """Generate cosine drift and AUROC plots for a single experiment."""
    output = Path(output_dir)
    plot_cosine_drift(str(csv_path), str(output / "cosine_drift.png"), concepts)
    plot_auroc_degradation(str(csv_path), str(output / "auroc_degradation.png"), concepts)


# ---------------------------------------------------------------------------
# Multi-experiment comparative plots
# ---------------------------------------------------------------------------


def _load_multi_experiment(
    experiment_csvs: dict[str, str],
    concepts: Optional[Sequence[str]] = None,
) -> tuple[dict[str, list[dict]], list[str]]:
    """Load rows from multiple experiment CSVs. Returns (label->rows, concepts)."""
    all_rows = {}
    for label, csv_path in experiment_csvs.items():
        all_rows[label] = read_csv_rows(csv_path)
    if concepts is None:
        first_path = next(iter(experiment_csvs.values()))
        concepts = infer_concepts(first_path)
    return all_rows, list(concepts)


def plot_comparative_cosine_drift(
    experiment_csvs: dict[str, str],
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """One subplot per concept showing cosine drift curves from each experiment."""
    import matplotlib.pyplot as plt

    all_rows, concepts = _load_multi_experiment(experiment_csvs, concepts)
    n = len(concepts)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for idx, concept in enumerate(concepts):
        ax = axes[0, idx]
        column = f"{concept}_cosine_to_baseline"
        for label, rows in all_rows.items():
            steps = [to_float(r["step"]) for r in rows]
            values = [to_float(r[column]) for r in rows]
            ax.plot(steps, values, marker="o", markersize=3, label=label)

        ax.axhline(DRIFT_THRESHOLD, linestyle="--", color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cosine similarity to baseline")
        ax.set_title(concept.capitalize())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Concept direction drift across experiments", fontsize=13)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_drift_onset_table(
    experiment_csvs: dict[str, str],
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
    threshold: float = DRIFT_THRESHOLD,
):
    """Render a table image showing the step where cosine < threshold."""
    import matplotlib.pyplot as plt

    all_rows, concepts = _load_multi_experiment(experiment_csvs, concepts)
    labels = list(experiment_csvs.keys())

    cell_text = []
    for concept in concepts:
        row_cells = []
        column = f"{concept}_cosine_to_baseline"
        for label in labels:
            rows = all_rows[label]
            onset_step = None
            for r in rows:
                if to_float(r[column]) < threshold:
                    onset_step = int(to_float(r["step"]))
                    break
            row_cells.append(str(onset_step) if onset_step is not None else "--")
        cell_text.append(row_cells)

    fig, ax = plt.subplots(figsize=(max(4, 2 * len(labels)), max(2, 0.6 * len(concepts) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[c.capitalize() for c in concepts],
        colLabels=labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    ax.set_title(f"Drift onset (step where cosine < {threshold})", fontsize=12, pad=20)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_comparative_auroc(
    experiment_csvs: dict[str, str],
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """One subplot per concept showing fixed-direction AUROC from each experiment."""
    import matplotlib.pyplot as plt

    all_rows, concepts = _load_multi_experiment(experiment_csvs, concepts)
    n = len(concepts)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for idx, concept in enumerate(concepts):
        ax = axes[0, idx]
        column = f"{concept}_auroc_fixed"
        for label, rows in all_rows.items():
            steps = [to_float(r["step"]) for r in rows]
            values = [to_float(r[column]) for r in rows]
            ax.plot(steps, values, marker="o", markersize=3, label=label)

        ax.axhline(MIN_AUROC_DEFAULT, linestyle="--", color="black", linewidth=1, alpha=0.6)
        ax.set_xlabel("Training step")
        ax.set_ylabel("AUROC (fixed direction)")
        ax.set_title(concept.capitalize())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Fixed-direction AUROC across experiments", fontsize=13)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_comparative_cross_concept(
    experiment_csvs: dict[str, str],
    output_path: str,
):
    """Cross-concept coupling evolution per experiment, side by side."""
    import matplotlib.pyplot as plt

    all_rows = {}
    for label, csv_path in experiment_csvs.items():
        all_rows[label] = read_csv_rows(csv_path)

    labels = list(experiment_csvs.keys())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for idx, label in enumerate(labels):
        ax = axes[0, idx]
        rows = all_rows[label]
        if not rows:
            continue
        cross_cols = sorted(k for k in rows[0].keys() if k.startswith("cross_") and k.endswith("_cosine"))
        steps = [to_float(r["step"]) for r in rows]
        for col in cross_cols:
            pair_label = col.replace("cross_", "").replace("_cosine", "").replace("_", " vs ", 1)
            ax.plot(steps, [to_float(r[col]) for r in rows], marker="o", markersize=3, label=pair_label)

        ax.axhline(0, linestyle="--", color="black", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Cross-concept cosine")
        ax.set_title(label)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Cross-concept coupling during fine-tuning", fontsize=13)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_comparative_norm_dynamics(
    experiment_csvs: dict[str, str],
    output_path: str,
    concepts: Optional[Sequence[str]] = None,
):
    """One subplot per concept showing direction norm evolution from each experiment."""
    import matplotlib.pyplot as plt

    all_rows, concepts = _load_multi_experiment(experiment_csvs, concepts)
    n = len(concepts)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for idx, concept in enumerate(concepts):
        ax = axes[0, idx]
        column = f"{concept}_direction_norm"
        for label, rows in all_rows.items():
            if column not in rows[0]:
                continue
            steps = [to_float(r["step"]) for r in rows]
            values = [to_float(r[column]) for r in rows]
            ax.plot(steps, values, marker="o", markersize=3, label=label)

        ax.set_xlabel("Training step")
        ax.set_ylabel("Direction norm")
        ax.set_title(concept.capitalize())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Direction norm dynamics across experiments", fontsize=13)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def plot_experiment_comparison(
    experiment_csvs: dict[str, str],
    output_dir: str,
    concepts: Optional[Sequence[str]] = None,
):
    """Generate all comparative plots for a multi-experiment analysis."""
    output = Path(output_dir)
    plot_comparative_cosine_drift(experiment_csvs, str(output / "comparative_cosine_drift.png"), concepts)
    plot_drift_onset_table(experiment_csvs, str(output / "drift_onset_table.png"), concepts)
    plot_comparative_auroc(experiment_csvs, str(output / "comparative_auroc.png"), concepts)
    plot_comparative_cross_concept(experiment_csvs, str(output / "comparative_cross_concept.png"))
    plot_comparative_norm_dynamics(experiment_csvs, str(output / "comparative_norm_dynamics.png"), concepts)
