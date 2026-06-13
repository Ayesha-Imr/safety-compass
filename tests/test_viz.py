import csv
from pathlib import Path

from safety_compass.viz import (
    infer_concepts,
    plot_all_outputs,
    plot_cross_concept_evolution,
    plot_metric_heatmap,
)


def _write_fixture_csv(path):
    fieldnames = [
        "step", "epoch", "elapsed_seconds",
        "refusal_cosine_to_baseline", "refusal_auroc_fixed",
        "refusal_auroc_current", "refusal_direction_norm",
        "refusal_measurement_seconds",
        "sycophancy_cosine_to_baseline", "sycophancy_auroc_fixed",
        "sycophancy_auroc_current", "sycophancy_direction_norm",
        "sycophancy_measurement_seconds",
        "cross_refusal_sycophancy_cosine",
    ]
    rows = [
        {
            "step": "0", "epoch": "0.0", "elapsed_seconds": "0.0",
            "refusal_cosine_to_baseline": "1.0", "refusal_auroc_fixed": "1.0",
            "refusal_auroc_current": "1.0", "refusal_direction_norm": "130.0",
            "refusal_measurement_seconds": "5.0",
            "sycophancy_cosine_to_baseline": "1.0", "sycophancy_auroc_fixed": "1.0",
            "sycophancy_auroc_current": "1.0", "sycophancy_direction_norm": "6.0",
            "sycophancy_measurement_seconds": "5.0",
            "cross_refusal_sycophancy_cosine": "0.01",
        },
        {
            "step": "50", "epoch": "0.5", "elapsed_seconds": "100.0",
            "refusal_cosine_to_baseline": "0.85", "refusal_auroc_fixed": "0.95",
            "refusal_auroc_current": "0.98", "refusal_direction_norm": "125.0",
            "refusal_measurement_seconds": "5.0",
            "sycophancy_cosine_to_baseline": "0.60", "sycophancy_auroc_fixed": "1.0",
            "sycophancy_auroc_current": "1.0", "sycophancy_direction_norm": "5.5",
            "sycophancy_measurement_seconds": "5.0",
            "cross_refusal_sycophancy_cosine": "0.05",
        },
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_infer_concepts_from_csv(tmp_path):
    csv_path = str(tmp_path / "drift.csv")
    _write_fixture_csv(csv_path)
    concepts = infer_concepts(csv_path)
    assert concepts == ["refusal", "sycophancy"]


def test_plot_metric_heatmap_creates_file(tmp_path):
    csv_path = str(tmp_path / "drift.csv")
    _write_fixture_csv(csv_path)
    out = str(tmp_path / "heatmap.png")
    plot_metric_heatmap(csv_path, out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_plot_cross_concept_evolution_creates_file(tmp_path):
    csv_path = str(tmp_path / "drift.csv")
    _write_fixture_csv(csv_path)
    out = str(tmp_path / "cross.png")
    plot_cross_concept_evolution(csv_path, out)
    assert Path(out).exists()
    assert Path(out).stat().st_size > 0


def test_plot_all_outputs_creates_files(tmp_path):
    csv_path = str(tmp_path / "drift.csv")
    _write_fixture_csv(csv_path)
    plot_all_outputs(csv_path, str(tmp_path))
    assert (tmp_path / "cosine_drift.png").exists()
    assert (tmp_path / "auroc_degradation.png").exists()
    assert (tmp_path / "metric_heatmap.png").exists()
    assert (tmp_path / "cross_concept_evolution.png").exists()
