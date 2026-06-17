#!/usr/bin/env python3
"""Extract concept directions, validate AUROCs, and save baseline artifacts.

Usage:
    python scripts/extract_directions.py \
        --experiment-config configs/experiments/alpaca_qlora.yaml \
        --output-dir results/baselines/

    python scripts/extract_directions.py \
        --experiment-config configs/experiments/alpaca_qlora.yaml \
        --output-dir results/baselines/ \
        --concepts refusal,deception

    python scripts/extract_directions.py --list-strategies
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safety_compass.concept import ConceptDirectionExtractor, _STRATEGIES
from safety_compass.config import load_experiment_config
from safety_compass.monitor import ConceptBaseline, save_baselines_to_dir
from safety_compass.utils import MIN_AUROC_DEFAULT, load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Extract concept directions and save baseline artifacts"
    )
    parser.add_argument(
        "--experiment-config",
        required="--list-strategies" not in sys.argv,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="results/baselines",
        help="Directory for baseline artifacts (default: results/baselines)",
    )
    parser.add_argument(
        "--concepts",
        default=None,
        help="Comma-separated concept names to extract (default: all from config)",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Override model name from config (e.g., local checkpoint path)",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for resolving relative paths in configs",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List registered pairing strategies and exit",
    )
    args = parser.parse_args()

    if args.list_strategies:
        print("Registered pairing strategies:")
        for name in sorted(_STRATEGIES.keys()):
            fn = _STRATEGIES[name]
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {name}: {doc}")
        return

    config_path = Path(args.experiment_config)
    base_dir = Path(args.base_dir) if args.base_dir else config_path.parent
    resolved = load_experiment_config(config_path, base_dir=base_dir)

    model_config = resolved["_resolved_model_config"]
    concept_configs = resolved["_resolved_concept_configs"]

    concept_filter = None
    if args.concepts:
        concept_filter = {c.strip() for c in args.concepts.split(",")}

    model, tokenizer = load_model_and_tokenizer(
        model_config, args.model_name_or_path,
        output_hidden_states=True, eval_mode=True,
    )

    import numpy as np
    import torch

    concept_layers = {}
    for entry in resolved["concepts"]:
        if entry.get("best_layer") is not None:
            concept_layers[entry["name"]] = entry["best_layer"]

    baselines = {}
    extraction_results = {}
    total_start = time.time()

    for concept_cfg in concept_configs:
        name = concept_cfg["name"]
        if concept_filter and name not in concept_filter:
            continue

        print(f"\n{'='*50}")
        print(f"Concept: {name}")
        print(f"{'='*50}")
        concept_start = time.time()

        extractor = ConceptDirectionExtractor(
            model=model,
            tokenizer=tokenizer,
            concept_config=concept_cfg,
            model_config=model_config,
        )
        extractor.load_pairs(base_dir=str(base_dir))

        layer = concept_layers.get(name, concept_cfg.get("best_layer"))

        if layer is not None:
            print(f"  Extracting direction at layer {layer}...")
            raw_dir = extractor.extract_direction(layer, split="train", normalize=False)
            direction_norm = float(np.linalg.norm(raw_dir))
            norm_dir = raw_dir / direction_norm if direction_norm > 0 else raw_dir
            auroc = float(extractor.validate_direction(norm_dir, layer))

            extraction_results[name] = {
                "best_layer": layer,
                "best_auroc": auroc,
                "direction_norm": direction_norm,
                "method": "fixed_layer",
            }
        else:
            print("  Running full layer sweep...")
            result = extractor.find_best_layer()
            layer = result["best_layer"]
            auroc = result["best_auroc"]
            norm_dir = result["best_direction"]
            direction_norm = result["layer_results"][layer]["direction_norm"]

            extraction_results[name] = {
                "best_layer": layer,
                "best_auroc": auroc,
                "direction_norm": direction_norm,
                "method": "layer_sweep",
                "layer_results": {
                    str(k): v for k, v in result["layer_results"].items()
                },
            }

        baselines[name] = ConceptBaseline(
            name=name,
            layer=layer,
            direction=np.asarray(norm_dir, dtype=np.float32),
            baseline_auroc=auroc,
            direction_norm=direction_norm,
        )

        concept_time = time.time() - concept_start
        status = "PASS" if auroc >= concept_cfg.get("min_auroc", MIN_AUROC_DEFAULT) else "FAIL"
        print(f"  Layer: {layer}, AUROC: {auroc:.4f}, Norm: {direction_norm:.4f} [{status}]")
        print(f"  Time: {concept_time:.1f}s")

        extractor.clear_cache()
        torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    save_baselines_to_dir(
        baselines, output_dir, model_name=model_config["model_name"]
    )

    results_path = output_dir / "extraction_results.json"
    with open(results_path, "w") as f:
        json.dump(extraction_results, f, indent=2)

    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print("EXTRACTION RESULTS")
    print(f"{'='*60}")
    print(f"\n{'Concept':<15} {'Layer':<8} {'AUROC':<10} {'Norm':<12} {'Status'}")
    print("-" * 55)
    for name, bl in baselines.items():
        min_auroc = MIN_AUROC_DEFAULT
        for c in concept_configs:
            if c["name"] == name:
                min_auroc = c.get("min_auroc", MIN_AUROC_DEFAULT)
                break
        status = "PASS" if bl.baseline_auroc >= min_auroc else "FAIL"
        print(f"{name:<15} {bl.layer:<8} {bl.baseline_auroc:<10.4f} {bl.direction_norm:<12.4f} {status}")

    print(f"\nArtifacts saved to: {output_dir}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
