from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import yaml

from safety_compass.concept import _STRATEGIES


class SafetyCompassConfigError(ValueError):
    """Raised when a configuration file is invalid or incomplete."""

    pass


def load_yaml(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.exists():
        raise SafetyCompassConfigError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SafetyCompassConfigError(f"Expected a YAML mapping in {path}, got {type(data).__name__}")
    return data


def _require_field(cfg: dict, field: str, expected_type: type, context: str) -> None:
    if field not in cfg:
        raise SafetyCompassConfigError(f"Missing required field '{field}' in {context}")
    if not isinstance(cfg[field], expected_type):
        raise SafetyCompassConfigError(
            f"Field '{field}' in {context} must be {expected_type.__name__}, "
            f"got {type(cfg[field]).__name__}"
        )


def _require_positive_int(cfg: dict, field: str, context: str) -> None:
    _require_field(cfg, field, int, context)
    if cfg[field] <= 0:
        raise SafetyCompassConfigError(
            f"Field '{field}' in {context} must be a positive integer, got {cfg[field]}"
        )


_CONCEPT_REQUIRED = {"name", "pairing_strategy", "contrastive_pairs_file"}
_CONCEPT_OPTIONAL = {
    "description": None,
    "extraction_split": "train",
    "validation_split": "val",
    "min_auroc": 0.80,
    "system_prompt": None,
    "positive_system_prompt": None,
    "negative_system_prompt": None,
    "best_layer": None,
}
_CONCEPT_KNOWN = _CONCEPT_REQUIRED | set(_CONCEPT_OPTIONAL)


def validate_concept_config(
    cfg: dict,
    base_dir: Optional[Union[str, Path]] = None,
) -> dict:
    context = "concept config"
    validated = dict(cfg)

    for field in _CONCEPT_REQUIRED:
        _require_field(validated, field, str, context)

    strategy = validated["pairing_strategy"]
    if strategy not in _STRATEGIES:
        available = ", ".join(sorted(_STRATEGIES.keys()))
        raise SafetyCompassConfigError(
            f"Unknown pairing strategy '{strategy}' in {context}. "
            f"Available: {available}. Register new strategies with @register_strategy('name')."
        )

    if base_dir is not None:
        pairs_path = Path(base_dir) / validated["contrastive_pairs_file"]
        if not pairs_path.exists():
            raise SafetyCompassConfigError(
                f"Contrastive pairs file not found: {pairs_path}. "
                "Run `python scripts/prepare_contrastive_pairs.py` first."
            )

    for key, default in _CONCEPT_OPTIONAL.items():
        validated.setdefault(key, default)

    if validated.get("best_layer") is not None:
        if not isinstance(validated["best_layer"], int) or validated["best_layer"] <= 0:
            raise SafetyCompassConfigError(
                f"Field 'best_layer' in {context} must be a positive integer, "
                f"got {validated['best_layer']}"
            )

    unknown = set(validated.keys()) - _CONCEPT_KNOWN
    if unknown:
        warnings.warn(
            f"Unknown fields in {context}: {sorted(unknown)}. "
            "These will be ignored.",
            stacklevel=2,
        )

    return validated


_MODEL_REQUIRED = {"model_name", "num_layers", "hidden_dim"}
_MODEL_OPTIONAL = {
    "extraction_batch_size": 4,
    "extraction_dtype": "float16",
    "quantization": None,
    "double_quant": True,
    "attn_implementation": "eager",
    "enable_thinking": False,
    "max_seq_length": 512,
}
_MODEL_KNOWN = _MODEL_REQUIRED | set(_MODEL_OPTIONAL)


def validate_model_config(cfg: dict) -> dict:
    context = "model config"
    validated = dict(cfg)

    _require_field(validated, "model_name", str, context)
    _require_positive_int(validated, "num_layers", context)
    _require_positive_int(validated, "hidden_dim", context)

    for key, default in _MODEL_OPTIONAL.items():
        validated.setdefault(key, default)

    unknown = set(validated.keys()) - _MODEL_KNOWN
    if unknown:
        warnings.warn(
            f"Unknown fields in {context}: {sorted(unknown)}. "
            "These will be ignored.",
            stacklevel=2,
        )

    return validated


def validate_experiment_config(
    cfg: dict,
    base_dir: Optional[Union[str, Path]] = None,
) -> dict:
    context = "experiment config"
    validated = dict(cfg)

    _require_field(validated, "model_config_file", str, context)
    _require_field(validated, "concepts", list, context)

    if not validated["concepts"]:
        raise SafetyCompassConfigError(f"'concepts' list in {context} must not be empty")

    names = []
    for i, concept_entry in enumerate(validated["concepts"]):
        entry_ctx = f"{context} concepts[{i}]"
        if not isinstance(concept_entry, dict):
            raise SafetyCompassConfigError(f"{entry_ctx} must be a mapping, got {type(concept_entry).__name__}")
        _require_field(concept_entry, "name", str, entry_ctx)
        _require_field(concept_entry, "config_file", str, entry_ctx)

        if concept_entry["name"] in names:
            raise SafetyCompassConfigError(
                f"Duplicate concept name '{concept_entry['name']}' in {context}"
            )
        names.append(concept_entry["name"])

        if "best_layer" in concept_entry and concept_entry["best_layer"] is not None:
            bl = concept_entry["best_layer"]
            if not isinstance(bl, int) or bl <= 0:
                raise SafetyCompassConfigError(
                    f"Field 'best_layer' in {entry_ctx} must be a positive integer, got {bl}"
                )

    return validated


def load_experiment_config(
    path: Union[str, Path],
    base_dir: Optional[Union[str, Path]] = None,
) -> dict:
    path = Path(path)
    if base_dir is None:
        base_dir = Path.cwd()
    base_dir = Path(base_dir)

    cfg = load_yaml(path)
    validated = validate_experiment_config(cfg, base_dir=base_dir)

    model_config_path = base_dir / validated["model_config_file"]
    model_cfg = load_yaml(model_config_path)
    resolved_model = validate_model_config(model_cfg)
    validated["_resolved_model_config"] = resolved_model

    resolved_concepts = []
    for concept_entry in validated["concepts"]:
        concept_config_path = base_dir / concept_entry["config_file"]
        concept_cfg = load_yaml(concept_config_path)
        concept_validated = validate_concept_config(concept_cfg, base_dir=base_dir)

        if "best_layer" in concept_entry and concept_entry["best_layer"] is not None:
            concept_validated["best_layer"] = concept_entry["best_layer"]
            num_layers = resolved_model["num_layers"]
            if concept_validated["best_layer"] > num_layers:
                raise SafetyCompassConfigError(
                    f"best_layer ({concept_validated['best_layer']}) for concept "
                    f"'{concept_entry['name']}' exceeds model num_layers ({num_layers})"
                )

        resolved_concepts.append(concept_validated)

    validated["_resolved_concept_configs"] = resolved_concepts
    return validated
