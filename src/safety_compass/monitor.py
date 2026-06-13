from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
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
