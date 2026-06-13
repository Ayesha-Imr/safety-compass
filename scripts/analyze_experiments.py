#!/usr/bin/env python3
"""Analyze Phase 3 experiment results and produce publication-ready figures.

Usage:
    python scripts/analyze_experiments.py \
        --results-dir results/phase3/ \
        --output-dir results/phase3/analysis/

    python scripts/analyze_experiments.py \
        --csv "Alpaca=results/phase3/exp1/drift_log.csv" \
              "Dolly=results/phase3/exp2/drift_log.csv" \
              "Code Alpaca=results/phase3/exp3/drift_log.csv" \
        --output-dir results/phase3/analysis/
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


DEFAULT_EXPERIMENT_DIRS = {
    "Alpaca": "exp1",
    "Dolly": "exp2",
    "Code Alpaca": "exp3",
}

DRIFT_THRESHOLD = 0.95


def _find_csv(directory: Path) -> Path:
    """Find the drift log CSV in a results directory."""
    candidates = list(directory.glob("*drift_log*.csv")) + list(directory.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in {directory}")
    return candidates[0]


def _read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def compute_drift_onset(rows: list[dict], concept: str, threshold: float = DRIFT_THRESHOLD):
    """Return the first step where cosine_to_baseline < threshold, or None."""
    column = f"{concept}_cosine_to_baseline"
    for row in rows:
        if _to_float(row[column]) < threshold:
            return int(_to_float(row["step"]))
    return None


def compute_experiment_summary(rows: list[dict], concepts: list[str]) -> dict:
    """Compute per-concept summary statistics for a single experiment."""
    summary = {}
    for concept in concepts:
        cosine_col = f"{concept}_cosine_to_baseline"
        auroc_col = f"{concept}_auroc_fixed"
        norm_col = f"{concept}_direction_norm"

        cosines = [_to_float(r[cosine_col]) for r in rows]
        aurocs = [_to_float(r[auroc_col]) for r in rows]

        min_cos = min(cosines)
        min_cos_idx = cosines.index(min_cos)

        summary[concept] = {
            "min_cosine": min_cos,
            "final_cosine": cosines[-1],
            "step_at_min_cosine": int(_to_float(rows[min_cos_idx]["step"])),
            "drift_onset_step": compute_drift_onset(rows, concept),
            "min_auroc_fixed": min(aurocs),
            "final_auroc_fixed": aurocs[-1],
        }

        if norm_col in rows[0]:
            norms = [_to_float(r[norm_col]) for r in rows]
            summary[concept]["initial_norm"] = norms[0]
            summary[concept]["final_norm"] = norms[-1]
            summary[concept]["norm_change_pct"] = (
                (norms[-1] - norms[0]) / norms[0] * 100 if norms[0] != 0 else 0
            )

    return summary


def compute_fragility_ranking(summary: dict) -> list[str]:
    """Rank concepts by fragility (most fragile first) based on min cosine."""
    return sorted(summary.keys(), key=lambda c: summary[c]["min_cosine"])


def print_drift_onset_table(all_summaries: dict[str, dict], concepts: list[str]):
    """Print the drift onset table to stdout."""
    labels = list(all_summaries.keys())
    header = f"{'Concept':<15}" + "".join(f"{label:>20}" for label in labels)
    print("\nDrift Onset Table (step where cosine < 0.95)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for concept in concepts:
        row = f"{concept.capitalize():<15}"
        for label in labels:
            onset = all_summaries[label][concept]["drift_onset_step"]
            row += f"{str(onset) if onset is not None else '--':>20}"
        print(row)
    print()


def print_fragility_rankings(all_summaries: dict[str, dict]):
    """Print fragility ranking per experiment."""
    print("Fragility Rankings (most fragile first)")
    print("=" * 50)
    for label, summary in all_summaries.items():
        ranking = compute_fragility_ranking(summary)
        ranked_str = " > ".join(
            f"{c} ({summary[c]['min_cosine']:.4f})" for c in ranking
        )
        print(f"  {label}: {ranked_str}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Phase 3 experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing exp1/, exp2/, exp3/ subdirectories",
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        default=None,
        help="Explicit CSV paths as label=path pairs (e.g. 'Alpaca=path/to/csv')",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--concepts",
        default=None,
        help="Comma-separated concept names (auto-detected from CSV if omitted)",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also save figures as PDF",
    )
    args = parser.parse_args()

    from safety_compass.viz import (
        infer_concepts,
        plot_all_outputs,
        plot_phase3_comparison,
    )

    experiment_csvs: dict[str, str] = {}

    if args.csv:
        for entry in args.csv:
            if "=" not in entry:
                parser.error(f"CSV entries must be label=path format, got: {entry}")
            label, path = entry.split("=", 1)
            experiment_csvs[label.strip()] = path.strip()
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        for label, subdir in DEFAULT_EXPERIMENT_DIRS.items():
            exp_dir = results_dir / subdir
            if exp_dir.exists():
                csv_path = _find_csv(exp_dir)
                experiment_csvs[label] = str(csv_path)
                print(f"  Found {label}: {csv_path}")
            else:
                print(f"  Skipping {label}: {exp_dir} not found")
    else:
        parser.error("Provide either --results-dir or --csv")

    if not experiment_csvs:
        print("ERROR: No experiment CSVs found.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_csv = next(iter(experiment_csvs.values()))
    concepts = args.concepts.split(",") if args.concepts else infer_concepts(first_csv)
    print(f"  Concepts: {concepts}")

    print("\nComputing experiment summaries...")
    all_summaries = {}
    for label, csv_path in experiment_csvs.items():
        rows = _read_rows(csv_path)
        all_summaries[label] = compute_experiment_summary(rows, concepts)

    print_drift_onset_table(all_summaries, concepts)
    print_fragility_rankings(all_summaries)

    analysis = {
        "experiments": {
            label: {
                "csv_path": csv_path,
                "summary": all_summaries[label],
                "fragility_ranking": compute_fragility_ranking(all_summaries[label]),
            }
            for label, csv_path in experiment_csvs.items()
        },
        "concepts": concepts,
        "drift_threshold": DRIFT_THRESHOLD,
    }

    analysis_path = output_dir / "phase3_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {analysis_path}")

    print("\nGenerating per-experiment plots...")
    for label, csv_path in experiment_csvs.items():
        label_dir = output_dir / label.lower().replace(" ", "_")
        label_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_all_outputs(csv_path, str(label_dir), concepts=concepts)
            print(f"  {label}: {label_dir}")
        except Exception as e:
            print(f"  {label}: plot generation failed: {e}")

    print("\nGenerating comparative plots...")
    try:
        plot_phase3_comparison(experiment_csvs, str(output_dir), concepts=concepts)
        print(f"  Comparative plots saved to: {output_dir}")
    except Exception as e:
        print(f"  Comparative plot generation failed: {e}")

    if args.pdf:
        print("\nGenerating PDF versions...")
        for png_path in output_dir.rglob("*.png"):
            pdf_path = png_path.with_suffix(".pdf")
            try:
                from matplotlib.image import imread
                import matplotlib.pyplot as plt

                img = imread(str(png_path))
                fig, ax = plt.subplots(figsize=(img.shape[1] / 160, img.shape[0] / 160))
                ax.imshow(img)
                ax.axis("off")
                fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  PDF conversion failed for {png_path.name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
