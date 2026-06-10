#!/usr/bin/env python3
"""Generate contrastive prompt pairs for safety concept direction extraction.

Uses the auto-discovering data_sources registry — each concept is a self-contained
module in src/safety_compass/data_sources/. No changes to this script are needed
when adding new concepts.

Usage:
    pip install datasets huggingface-hub
    python scripts/prepare_contrastive_pairs.py --output-dir data/contrastive_pairs/
    python scripts/prepare_contrastive_pairs.py --concepts refusal,deception --seed 123
    python scripts/prepare_contrastive_pairs.py --list
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safety_compass.data_sources import generate_pairs, list_data_sources


def write_jsonl(entries: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(entries)} entries to {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate contrastive prompt pairs")
    parser.add_argument(
        "--output-dir",
        default="data/contrastive_pairs",
        help="Output directory for JSONL files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--concepts",
        default=None,
        help="Comma-separated list of concepts (default: all registered)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available data sources and exit",
    )
    args = parser.parse_args()

    available = list_data_sources()

    if args.list:
        print("Available data sources:")
        for name, desc in available.items():
            print(f"  {name}: {desc}")
        return

    concepts = (
        [c.strip() for c in args.concepts.split(",")]
        if args.concepts
        else list(available.keys())
    )

    output_dir = Path(args.output_dir)

    for concept_name in concepts:
        if concept_name not in available:
            print(f"Unknown concept: {concept_name}. Available: {', '.join(available.keys())}")
            continue

        print(f"Preparing {concept_name}...")
        pairs = generate_pairs(concept_name, seed=args.seed)
        write_jsonl(pairs, output_dir / f"{concept_name}.jsonl")

        n_train = sum(1 for p in pairs if p.get("split") == "train")
        n_val = sum(1 for p in pairs if p.get("split") == "val")
        print(f"  {concept_name}: {n_train} train, {n_val} val")

    print("Done.")


if __name__ == "__main__":
    main()
