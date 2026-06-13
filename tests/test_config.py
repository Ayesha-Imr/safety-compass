import json
import warnings

import pytest
import yaml

from safety_compass.config import (
    SafetyCompassConfigError,
    load_experiment_config,
    validate_concept_config,
    validate_experiment_config,
    validate_model_config,
)


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _minimal_concept_config(**overrides):
    cfg = {
        "name": "refusal",
        "pairing_strategy": "arditi",
        "contrastive_pairs_file": "data/pairs.jsonl",
    }
    cfg.update(overrides)
    return cfg


def _minimal_model_config(**overrides):
    cfg = {
        "model_name": "test/model",
        "num_layers": 12,
        "hidden_dim": 768,
    }
    cfg.update(overrides)
    return cfg


# --- Concept config tests ---


def test_valid_concept_config_passes():
    result = validate_concept_config(_minimal_concept_config())
    assert result["name"] == "refusal"
    assert result["pairing_strategy"] == "arditi"
    assert result["extraction_split"] == "train"
    assert result["min_auroc"] == 0.80


def test_concept_config_missing_name_raises():
    cfg = _minimal_concept_config()
    del cfg["name"]
    with pytest.raises(SafetyCompassConfigError, match="Missing required field 'name'"):
        validate_concept_config(cfg)


def test_concept_config_missing_pairs_file_raises():
    cfg = _minimal_concept_config()
    del cfg["contrastive_pairs_file"]
    with pytest.raises(SafetyCompassConfigError, match="contrastive_pairs_file"):
        validate_concept_config(cfg)


def test_concept_config_invalid_strategy_raises():
    cfg = _minimal_concept_config(pairing_strategy="nonexistent")
    with pytest.raises(SafetyCompassConfigError, match="Unknown pairing strategy"):
        validate_concept_config(cfg)


def test_concept_config_pairs_file_not_found_raises(tmp_path):
    cfg = _minimal_concept_config(contrastive_pairs_file="missing.jsonl")
    with pytest.raises(SafetyCompassConfigError, match="Contrastive pairs file not found"):
        validate_concept_config(cfg, base_dir=tmp_path)


def test_concept_config_pairs_file_exists_passes(tmp_path):
    pairs_path = tmp_path / "data" / "pairs.jsonl"
    _write_jsonl(pairs_path, [{"dummy": True}])
    cfg = _minimal_concept_config(contrastive_pairs_file="data/pairs.jsonl")
    result = validate_concept_config(cfg, base_dir=tmp_path)
    assert result["name"] == "refusal"


def test_concept_config_defaults_applied():
    result = validate_concept_config(_minimal_concept_config())
    assert result["extraction_split"] == "train"
    assert result["validation_split"] == "val"
    assert result["min_auroc"] == 0.80
    assert result["description"] is None
    assert result["best_layer"] is None


def test_concept_config_invalid_best_layer_raises():
    cfg = _minimal_concept_config(best_layer=-1)
    with pytest.raises(SafetyCompassConfigError, match="best_layer"):
        validate_concept_config(cfg)


def test_concept_config_unknown_fields_warns():
    cfg = _minimal_concept_config(future_field="value")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_concept_config(cfg)
        assert len(w) == 1
        assert "Unknown fields" in str(w[0].message)
        assert "future_field" in str(w[0].message)


# --- Model config tests ---


def test_valid_model_config_passes():
    result = validate_model_config(_minimal_model_config())
    assert result["model_name"] == "test/model"
    assert result["num_layers"] == 12
    assert result["extraction_batch_size"] == 4


def test_model_config_missing_num_layers_raises():
    cfg = _minimal_model_config()
    del cfg["num_layers"]
    with pytest.raises(SafetyCompassConfigError, match="num_layers"):
        validate_model_config(cfg)


def test_model_config_negative_num_layers_raises():
    cfg = _minimal_model_config(num_layers=-1)
    with pytest.raises(SafetyCompassConfigError, match="positive integer"):
        validate_model_config(cfg)


# --- Experiment config tests ---


def test_valid_experiment_config_passes():
    cfg = {
        "model_config_file": "models/test.yaml",
        "concepts": [
            {"name": "refusal", "config_file": "concepts/refusal.yaml", "best_layer": 10},
        ],
    }
    result = validate_experiment_config(cfg)
    assert result["concepts"][0]["name"] == "refusal"


def test_experiment_config_duplicate_concept_names_raises():
    cfg = {
        "model_config_file": "models/test.yaml",
        "concepts": [
            {"name": "refusal", "config_file": "concepts/refusal.yaml"},
            {"name": "refusal", "config_file": "concepts/refusal2.yaml"},
        ],
    }
    with pytest.raises(SafetyCompassConfigError, match="Duplicate concept name"):
        validate_experiment_config(cfg)


def test_experiment_config_empty_concepts_raises():
    cfg = {
        "model_config_file": "models/test.yaml",
        "concepts": [],
    }
    with pytest.raises(SafetyCompassConfigError, match="must not be empty"):
        validate_experiment_config(cfg)


def test_experiment_config_best_layer_exceeds_num_layers_raises(tmp_path):
    model_cfg = _minimal_model_config(num_layers=12)
    _write_yaml(tmp_path / "model.yaml", model_cfg)

    concept_cfg = _minimal_concept_config()
    _write_yaml(tmp_path / "concept.yaml", concept_cfg)
    _write_jsonl(tmp_path / "data" / "pairs.jsonl", [{"dummy": True}])

    exp_cfg = {
        "model_config_file": "model.yaml",
        "concepts": [
            {"name": "refusal", "config_file": "concept.yaml", "best_layer": 99},
        ],
    }
    _write_yaml(tmp_path / "experiment.yaml", exp_cfg)

    with pytest.raises(SafetyCompassConfigError, match="exceeds model num_layers"):
        load_experiment_config(tmp_path / "experiment.yaml", base_dir=tmp_path)


# --- load_experiment_config integration ---


def test_load_experiment_config_resolves_all_files(tmp_path):
    model_cfg = _minimal_model_config()
    _write_yaml(tmp_path / "models" / "test.yaml", model_cfg)

    concept_cfg = _minimal_concept_config()
    _write_yaml(tmp_path / "concepts" / "refusal.yaml", concept_cfg)
    _write_jsonl(tmp_path / "data" / "pairs.jsonl", [{"dummy": True}])

    exp_cfg = {
        "model_config_file": "models/test.yaml",
        "concepts": [
            {"name": "refusal", "config_file": "concepts/refusal.yaml", "best_layer": 5},
        ],
    }
    _write_yaml(tmp_path / "experiment.yaml", exp_cfg)

    result = load_experiment_config(tmp_path / "experiment.yaml", base_dir=tmp_path)
    assert result["_resolved_model_config"]["model_name"] == "test/model"
    assert result["_resolved_concept_configs"][0]["name"] == "refusal"
    assert result["_resolved_concept_configs"][0]["best_layer"] == 5
