from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from safety_compass.concept import ConceptDirectionExtractor


@dataclass
class ConceptBaseline:
    """Baseline direction metadata for one concept."""

    name: str
    layer: int
    direction: np.ndarray
    baseline_auroc: float
    direction_norm: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "layer": self.layer,
            "baseline_auroc": self.baseline_auroc,
            "direction_norm": self.direction_norm,
        }


class SafetyCompassMonitor:
    """Measure concept-direction drift for a model at training time.

    The monitor is intentionally training-loop agnostic. It owns concept extractors,
    baseline directions, and metric computation; callbacks or scripts decide when
    measurements happen.
    """

    def __init__(
        self,
        model,
        tokenizer,
        concept_configs: Union[Mapping[str, dict], Sequence[dict]],
        model_config: dict,
        base_dir: Optional[str] = None,
        concept_layers: Optional[Mapping[str, int]] = None,
        include_cross_concept_cosines: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = dict(model_config)
        self.base_dir = base_dir
        self.concept_layers = dict(concept_layers or {})
        self.include_cross_concept_cosines = include_cross_concept_cosines

        self.concept_configs = self._normalize_concept_configs(concept_configs)
        self.concept_names = [cfg["name"] for cfg in self.concept_configs]
        self.extractors = {}
        for cfg in self.concept_configs:
            extractor = ConceptDirectionExtractor(
                model=self.model,
                tokenizer=self.tokenizer,
                concept_config=cfg,
                model_config=self.model_config,
            )
            extractor.load_pairs(base_dir=self.base_dir)
            self.extractors[cfg["name"]] = extractor

        self.baselines: dict[str, ConceptBaseline] = {}
        self.started_at = time.perf_counter()

    @staticmethod
    def _normalize_concept_configs(concept_configs) -> list[dict]:
        if isinstance(concept_configs, Mapping):
            normalized = []
            for name, cfg in concept_configs.items():
                copied = dict(cfg)
                copied.setdefault("name", name)
                normalized.append(copied)
            return normalized
        return [dict(cfg) for cfg in concept_configs]

    @staticmethod
    def _unit_normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    @classmethod
    def from_config(
        cls,
        model,
        tokenizer,
        experiment_config: Union[str, Path, dict],
        *,
        base_dir: Optional[Union[str, Path]] = None,
        overrides: Optional[dict] = None,
    ) -> "SafetyCompassMonitor":
        from safety_compass.config import (
            SafetyCompassConfigError,
            load_experiment_config,
            validate_experiment_config,
            validate_model_config,
        )

        if isinstance(experiment_config, (str, Path)):
            resolved = load_experiment_config(experiment_config, base_dir=base_dir)
            if base_dir is None:
                base_dir = Path.cwd()
        elif isinstance(experiment_config, dict):
            if base_dir is None:
                raise SafetyCompassConfigError(
                    "base_dir is required when experiment_config is a dict"
                )
            resolved = dict(experiment_config)
            if "_resolved_model_config" not in resolved:
                resolved = validate_experiment_config(resolved, base_dir=base_dir)
                from safety_compass.config import load_yaml, validate_concept_config

                model_cfg = load_yaml(Path(base_dir) / resolved["model_config_file"])
                resolved["_resolved_model_config"] = validate_model_config(model_cfg)
                concepts = []
                for entry in resolved["concepts"]:
                    c = load_yaml(Path(base_dir) / entry["config_file"])
                    c = validate_concept_config(c, base_dir=base_dir)
                    if entry.get("best_layer") is not None:
                        c["best_layer"] = entry["best_layer"]
                    concepts.append(c)
                resolved["_resolved_concept_configs"] = concepts
        else:
            raise TypeError(
                f"experiment_config must be a str, Path, or dict, got {type(experiment_config)}"
            )

        if overrides:
            for section, values in overrides.items():
                if section in resolved and isinstance(resolved[section], dict):
                    resolved[section].update(values)
                else:
                    resolved[section] = values

        model_cfg = resolved["_resolved_model_config"]
        concept_configs = resolved["_resolved_concept_configs"]
        monitor_cfg = resolved.get("monitor", {})

        concept_layers = {}
        for c in concept_configs:
            if c.get("best_layer") is not None:
                concept_layers[c["name"]] = c["best_layer"]

        return cls(
            model=model,
            tokenizer=tokenizer,
            concept_configs=concept_configs,
            model_config=model_cfg,
            base_dir=str(base_dir) if base_dir else None,
            concept_layers=concept_layers,
            include_cross_concept_cosines=monitor_cfg.get(
                "include_cross_concept_cosines", True
            ),
        )

    def set_model(self, model):
        """Update the model reference used by all extractors."""
        self.model = model
        for extractor in self.extractors.values():
            extractor.set_model(model)

    @property
    def is_setup(self) -> bool:
        return bool(self.baselines)

    def setup(self, allow_layer_sweep: bool = True) -> dict[str, ConceptBaseline]:
        """Extract baseline directions and validate them once before training."""
        self.baselines = {}

        for cfg in self.concept_configs:
            name = cfg["name"]
            extractor = self.extractors[name]
            layer = self.concept_layers.get(name, cfg.get("best_layer"))
            extractor.clear_cache()

            if layer is None:
                if not allow_layer_sweep:
                    raise ValueError(
                        f"No best layer configured for concept '{name}'. "
                        "Pass concept_layers or enable allow_layer_sweep."
                    )
                result = extractor.find_best_layer()
                layer = int(result["best_layer"])
                direction = np.asarray(result["best_direction"])
                auroc = float(result["best_auroc"])
                direction_norm = float(result["layer_results"][layer]["direction_norm"])
            else:
                layer = int(layer)
                raw_direction = extractor.extract_direction(layer, split="train", normalize=False)
                direction_norm = float(np.linalg.norm(raw_direction))
                direction = self._unit_normalize(raw_direction)
                auroc = float(extractor.validate_direction(direction, layer))

            self.baselines[name] = ConceptBaseline(
                name=name,
                layer=layer,
                direction=direction,
                baseline_auroc=auroc,
                direction_norm=direction_norm,
            )
            extractor.clear_cache()

        return self.baselines

    def measure(self, step: int, epoch: Optional[float] = None) -> dict:
        """Measure current drift metrics for all configured concepts."""
        if not self.is_setup:
            self.setup()

        row = {
            "step": int(step),
            "epoch": "" if epoch is None else float(epoch),
            "elapsed_seconds": time.perf_counter() - self.started_at,
        }
        current_directions = {}

        for name in self.concept_names:
            extractor = self.extractors[name]
            baseline = self.baselines[name]
            concept_start = time.perf_counter()
            extractor.clear_cache()

            raw_direction = extractor.extract_direction(
                baseline.layer,
                split="train",
                normalize=False,
            )
            current_direction = self._unit_normalize(raw_direction)
            current_directions[name] = current_direction

            row[f"{name}_cosine_to_baseline"] = float(
                np.dot(current_direction, baseline.direction)
            )
            row[f"{name}_auroc_fixed"] = float(
                extractor.validate_direction(baseline.direction, baseline.layer)
            )
            row[f"{name}_auroc_current"] = float(
                extractor.validate_direction(current_direction, baseline.layer)
            )
            row[f"{name}_direction_norm"] = float(np.linalg.norm(raw_direction))
            row[f"{name}_measurement_seconds"] = time.perf_counter() - concept_start

            extractor.clear_cache()

        if self.include_cross_concept_cosines:
            row.update(self._cross_concept_cosines(current_directions))

        return row

    def _cross_concept_cosines(self, directions: Mapping[str, np.ndarray]) -> dict:
        results = {}
        for left, right in itertools.combinations(self.concept_names, 2):
            left_dir = directions[left]
            right_dir = directions[right]
            key = f"cross_{left}_{right}_cosine"
            if left_dir.shape != right_dir.shape:
                results[key] = float("nan")
            else:
                results[key] = float(np.dot(left_dir, right_dir))
        return results

    def baseline_summary(self) -> dict:
        return {name: baseline.to_dict() for name, baseline in self.baselines.items()}

    def save_baselines(self, output_dir: Union[str, Path]) -> None:
        if not self.baselines:
            raise RuntimeError("No baselines to save. Call setup() first.")
        save_baselines_to_dir(self.baselines, output_dir)

    def load_baselines(self, baselines_dir: Union[str, Path]) -> None:
        self.baselines = load_baselines_from_dir(baselines_dir)


def save_baselines_to_dir(
    baselines: dict[str, ConceptBaseline],
    output_dir: Union[str, Path],
    model_name: Optional[str] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    metadata = {}
    for name, bl in baselines.items():
        arrays[name] = np.asarray(bl.direction, dtype=np.float32)
        metadata[name] = {
            "layer": bl.layer,
            "baseline_auroc": bl.baseline_auroc,
            "direction_norm": bl.direction_norm,
        }

    np.savez(output_dir / "directions.npz", **arrays)

    meta = {"concepts": metadata}
    if model_name:
        meta["model_name"] = model_name
    with open(output_dir / "directions_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_baselines_from_dir(
    baselines_dir: Union[str, Path],
) -> dict[str, ConceptBaseline]:
    baselines_dir = Path(baselines_dir)
    npz_path = baselines_dir / "directions.npz"
    meta_path = baselines_dir / "directions_metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Baseline directions not found: {npz_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Baseline metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    concept_meta = meta["concepts"]
    data = np.load(npz_path, allow_pickle=False)

    baselines = {}
    for name, info in concept_meta.items():
        if name not in data:
            raise KeyError(f"Direction array for concept '{name}' not found in {npz_path}")
        baselines[name] = ConceptBaseline(
            name=name,
            layer=info["layer"],
            direction=data[name],
            baseline_auroc=info["baseline_auroc"],
            direction_norm=info["direction_norm"],
        )

    return baselines
