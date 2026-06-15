#!/usr/bin/env python3
"""Generate per-user Kaggle kernel metadata for pushing.

Detects the logged-in Kaggle user and creates a push directory under
/tmp/safety-compass-kernels/<phase>/ with the correct kernel-metadata.json
and a copy of the phase's script.py.

Usage:
    python kaggle/prepare_kernel.py phase0 --hf-token-dataset auto
    python kaggle/prepare_kernel.py all --username ayeshaimr --hf-token-dataset ayeshaimr/nsa-hf-token
"""

import argparse
import json
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PHASES = {
    "phase0": {
        "slug": "safety-compass-phase0",
        "title": "safety-compass-phase0",
        "script_dir": "kaggle/phase0_validate",
    },
    "phase1": {
        "slug": "safety-compass-phase1",
        "title": "safety-compass-phase1",
        "script_dir": "kaggle/phase1_drift",
    },
    "phase3": {
        "slug": "safety-compass-phase3",
        "title": "safety-compass-phase3",
        "script_dir": "kaggle/phase3_experiment",
    },
    "phase4": {
        "slug": "safety-compass-phase4",
        "title": "safety-compass-phase4",
        "script_dir": "kaggle/phase4_behavioral",
    },
}

OUTPUT_BASE = Path("/tmp/safety-compass-kernels")
AUTO_HF_TOKEN_DATASET_SLUG = "nsa-hf-token"


def get_kaggle_username() -> str:
    if "KAGGLE_USERNAME" in os.environ:
        return os.environ["KAGGLE_USERNAME"]

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json) as f:
            data = json.load(f)
            if "username" in data:
                return data["username"]

    raise RuntimeError(
        "Cannot determine Kaggle username. Set KAGGLE_USERNAME env var, "
        "configure ~/.kaggle/kaggle.json, or pass --username."
    )


def prepare_phase(
    phase_name: str,
    username: str,
    hf_token_dataset: str | None,
    phase3_experiment: int | None = None,
    phase4_experiment: int | None = None,
):
    if phase_name not in PHASES:
        print(f"  Skipping unknown phase: {phase_name}")
        return

    phase = PHASES[phase_name]
    output_dir = OUTPUT_BASE / phase_name
    output_dir.mkdir(parents=True, exist_ok=True)

    script_src = REPO_ROOT / phase["script_dir"] / "script.py"
    if not script_src.exists():
        print(f"  WARNING: {script_src} does not exist, skipping {phase_name}")
        return

    script_dst = output_dir / "script.py"
    shutil.copy(script_src, script_dst)
    if phase_name == "phase3" and phase3_experiment is not None:
        script_text = script_dst.read_text()
        old = 'EXPERIMENT_ID = int(os.environ.get("PHASE3_EXPERIMENT", "1"))'
        new = f'EXPERIMENT_ID = int(os.environ.get("PHASE3_EXPERIMENT", "{phase3_experiment}"))'
        if old not in script_text:
            raise RuntimeError(f"Could not patch Phase 3 experiment default in {script_dst}")
        script_dst.write_text(script_text.replace(old, new))
    if phase_name == "phase4" and phase4_experiment is not None:
        script_text = script_dst.read_text()
        old = 'EXPERIMENT_ID = int(os.environ.get("PHASE4_EXPERIMENT", "3"))'
        new = f'EXPERIMENT_ID = int(os.environ.get("PHASE4_EXPERIMENT", "{phase4_experiment}"))'
        if old not in script_text:
            raise RuntimeError(f"Could not patch Phase 4 experiment default in {script_dst}")
        script_dst.write_text(script_text.replace(old, new))

    dataset_sources = []
    if hf_token_dataset:
        resolved = hf_token_dataset
        if resolved == "auto":
            resolved = f"{username}/{AUTO_HF_TOKEN_DATASET_SLUG}"
        dataset_sources.append(resolved)

    metadata = {
        "id": f"{username}/{phase['slug']}",
        "title": phase["title"],
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
    }

    with open(output_dir / "kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Generated: {output_dir}/")
    print(f"    ID: {metadata['id']}")
    print(f"    Dataset sources: {metadata['dataset_sources']}")
    if phase_name == "phase3" and phase3_experiment is not None:
        print(f"    Phase 3 experiment baked into generated script: {phase3_experiment}")
    if phase_name == "phase4" and phase4_experiment is not None:
        print(f"    Phase 4 experiment baked into generated script: {phase4_experiment}")
    print("    Push with:")
    print(f"      kaggle kernels push -p {output_dir}/ --accelerator NvidiaTeslaT4")


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle kernel push directories")
    parser.add_argument(
        "phase",
        choices=list(PHASES.keys()) + ["all"],
        help="Which phase to prepare, or 'all'",
    )
    parser.add_argument("--username", default=None, help="Kaggle username (auto-detected if omitted)")
    parser.add_argument(
        "--hf-token-dataset",
        default=None,
        help=f"Kaggle dataset slug for HF token (use 'auto' for {{username}}/{AUTO_HF_TOKEN_DATASET_SLUG})",
    )
    parser.add_argument(
        "--phase3-experiment",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "For phase3, bake the selected experiment into the generated script. "
            "Defaults to PHASE3_EXPERIMENT if set."
        ),
    )
    parser.add_argument(
        "--phase4-experiment",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "For phase4, bake the selected experiment into the generated script. "
            "Defaults to PHASE4_EXPERIMENT if set."
        ),
    )
    args = parser.parse_args()

    username = args.username or get_kaggle_username()
    phase3_experiment = args.phase3_experiment
    if phase3_experiment is None and os.environ.get("PHASE3_EXPERIMENT"):
        phase3_experiment = int(os.environ["PHASE3_EXPERIMENT"])
        if phase3_experiment not in {1, 2, 3}:
            raise ValueError("PHASE3_EXPERIMENT must be one of 1, 2, or 3")
    phase4_experiment = args.phase4_experiment
    if phase4_experiment is None and os.environ.get("PHASE4_EXPERIMENT"):
        phase4_experiment = int(os.environ["PHASE4_EXPERIMENT"])
        if phase4_experiment not in {1, 2, 3}:
            raise ValueError("PHASE4_EXPERIMENT must be one of 1, 2, or 3")
    print(f"Kaggle user: {username}")

    phases = list(PHASES.keys()) if args.phase == "all" else [args.phase]
    for phase in phases:
        print(f"\nPreparing {phase}...")
        prepare_phase(
            phase,
            username,
            args.hf_token_dataset,
            phase3_experiment if phase == "phase3" else None,
            phase4_experiment if phase == "phase4" else None,
        )


if __name__ == "__main__":
    main()
