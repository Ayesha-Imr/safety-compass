import json

import pytest
import yaml

import safety_compass.monitor as monitor_module
from safety_compass.config import SafetyCompassConfigError
from safety_compass.monitor import SafetyCompassMonitor

from tests.test_monitor import FakeExtractor, FakeModel


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f)


def _write_jsonl(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _setup_configs(tmp_path):
    model_cfg = {
        "model_name": "test/model",
        "num_layers": 12,
        "hidden_dim": 768,
    }
    _write_yaml(tmp_path / "models" / "test.yaml", model_cfg)

    concept_cfg = {
        "name": "refusal",
        "pairing_strategy": "arditi",
        "contrastive_pairs_file": "data/pairs.jsonl",
    }
    _write_yaml(tmp_path / "concepts" / "refusal.yaml", concept_cfg)
    _write_jsonl(tmp_path / "data" / "pairs.jsonl", [{"dummy": True}])

    exp_cfg = {
        "model_config_file": "models/test.yaml",
        "concepts": [
            {"name": "refusal", "config_file": "concepts/refusal.yaml", "best_layer": 5},
        ],
        "monitor": {
            "measure_every_n_steps": 50,
            "include_cross_concept_cosines": True,
        },
    }
    _write_yaml(tmp_path / "experiment.yaml", exp_cfg)
    return tmp_path / "experiment.yaml"


def test_from_config_with_experiment_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor_module, "ConceptDirectionExtractor", FakeExtractor)
    exp_path = _setup_configs(tmp_path)

    monitor = SafetyCompassMonitor.from_config(
        model=FakeModel(),
        tokenizer=None,
        experiment_config=exp_path,
        base_dir=tmp_path,
    )

    assert monitor.concept_names == ["refusal"]
    assert monitor.concept_layers == {"refusal": 5}
    assert monitor.include_cross_concept_cosines is True


def test_from_config_with_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor_module, "ConceptDirectionExtractor", FakeExtractor)
    _setup_configs(tmp_path)

    exp_dict = {
        "model_config_file": "models/test.yaml",
        "concepts": [
            {"name": "refusal", "config_file": "concepts/refusal.yaml", "best_layer": 5},
        ],
    }

    monitor = SafetyCompassMonitor.from_config(
        model=FakeModel(),
        tokenizer=None,
        experiment_config=exp_dict,
        base_dir=tmp_path,
    )

    assert monitor.concept_names == ["refusal"]


def test_from_config_dict_without_base_dir_raises():
    with pytest.raises(SafetyCompassConfigError, match="base_dir is required"):
        SafetyCompassMonitor.from_config(
            model=None,
            tokenizer=None,
            experiment_config={"model_config_file": "x.yaml", "concepts": []},
        )


def test_from_config_overrides_apply(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor_module, "ConceptDirectionExtractor", FakeExtractor)
    exp_path = _setup_configs(tmp_path)

    monitor = SafetyCompassMonitor.from_config(
        model=FakeModel(),
        tokenizer=None,
        experiment_config=exp_path,
        base_dir=tmp_path,
        overrides={"monitor": {"include_cross_concept_cosines": False}},
    )

    assert monitor.include_cross_concept_cosines is False


def test_from_config_invalid_config_raises(tmp_path):
    bad_cfg = {"concepts": []}
    _write_yaml(tmp_path / "bad.yaml", bad_cfg)

    with pytest.raises(SafetyCompassConfigError):
        SafetyCompassMonitor.from_config(
            model=None,
            tokenizer=None,
            experiment_config=tmp_path / "bad.yaml",
        )
