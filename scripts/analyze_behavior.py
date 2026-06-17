#!/usr/bin/env python3
"""Analyze behavioral validation results across experiments.

Usage:
    python scripts/analyze_behavior.py \
        --results-dir results/behavioral/ \
        --output-dir results/behavioral/analysis

    python scripts/analyze_behavior.py \
        --summary "Code Alpaca=results/behavioral/exp3/behavior_summary.json" \
        --output-dir results/behavioral/analysis
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safety_compass.behavioral import build_drift_behavior_rows
from safety_compass.utils import DEFAULT_PLOT_DPI, DRIFT_THRESHOLD


DEFAULT_EXPERIMENT_DIRS = {
    "Alpaca": "exp1",
    "Dolly": "exp2",
    "Code Alpaca": "exp3",
}


def find_summary(directory: Path) -> Path:
    candidates = sorted(directory.glob("*behavior_summary.json"))
    if not candidates:
        raise FileNotFoundError(f"No behavior summary found in {directory}")
    return candidates[0]


def load_summary(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "experiment",
        "concept",
        "min_cosine",
        "final_cosine",
        "drift_onset_step",
        "final_auroc_fixed",
        "baseline_behavior_score",
        "final_behavior_score",
        "behavior_delta",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_drift_behavior_plot(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    plottable = [
        row
        for row in rows
        if row["min_cosine"] is not None and row["behavior_delta"] is not None
    ]
    if not plottable:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in plottable:
        ax.scatter(row["min_cosine"], row["behavior_delta"], s=80)
        ax.annotate(
            f"{row['experiment']}\n{row['concept']}",
            (row["min_cosine"], row["behavior_delta"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(0, color="0.6", linewidth=1, linestyle="--")
    ax.axvline(DRIFT_THRESHOLD, color="0.6", linewidth=1, linestyle="--")
    ax.set_xlabel("Minimum cosine to baseline")
    ax.set_ylabel("Behavior score delta")
    ax.set_title("Drift vs behavioral score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=DEFAULT_PLOT_DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze behavioral validation results")
    parser.add_argument("--results-dir", default=None, help="Directory containing exp1/, exp2/, exp3/")
    parser.add_argument(
        "--summary",
        nargs="+",
        default=None,
        help="Explicit summaries as label=path pairs",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs")
    args = parser.parse_args()

    summary_paths: dict[str, str] = {}
    if args.summary:
        for entry in args.summary:
            if "=" not in entry:
                parser.error(f"Summary entries must be label=path format, got: {entry}")
            label, path = entry.split("=", 1)
            summary_paths[label.strip()] = path.strip()
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        for label, subdir in DEFAULT_EXPERIMENT_DIRS.items():
            exp_dir = results_dir / subdir
            if exp_dir.exists():
                summary_path = find_summary(exp_dir)
                summary_paths[label] = str(summary_path)
                print(f"  Found {label}: {summary_path}")
            else:
                print(f"  Skipping {label}: {exp_dir} not found")
    else:
        parser.error("Provide either --results-dir or --summary")

    summaries = {label: load_summary(path) for label, path in summary_paths.items()}
    rows = build_drift_behavior_rows(summaries)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "drift_vs_behavior_table.csv"
    write_csv(table_path, rows)
    plot_path = output_dir / "drift_vs_behavior_plot.png"
    write_drift_behavior_plot(plot_path, rows)

    analysis_path = output_dir / "behavior_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(
            {
                "summaries": summary_paths,
                "drift_vs_behavior": rows,
            },
            f,
            indent=2,
        )

    print(f"\nWrote {table_path}")
    if plot_path.exists():
        print(f"Wrote {plot_path}")
    print(f"Wrote {analysis_path}")
    if rows:
        print("\nDrift vs behavior:")
        for row in rows:
            print(
                f"  {row['experiment']} / {row['concept']}: "
                f"min_cos={row['min_cosine']}, behavior_delta={row['behavior_delta']}"
            )


if __name__ == "__main__":
    main()
