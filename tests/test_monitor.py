import numpy as np

import safety_compass.monitor as monitor_module
from safety_compass.monitor import SafetyCompassMonitor


class FakeExtractor:
    def __init__(self, model, tokenizer, concept_config, model_config):
        self.model = model
        self.name = concept_config["name"]
        self.clear_count = 0

    def load_pairs(self, base_dir=None):
        return None

    def set_model(self, model):
        self.model = model
        self.clear_cache()

    def clear_cache(self):
        self.clear_count += 1

    def extract_direction(self, layer, split="train", normalize=True):
        vector = np.asarray(self.model.vectors[self.name], dtype=np.float32)
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        return vector

    def validate_direction(self, direction, layer):
        return 0.5 + 0.5 * float(np.clip(direction[0], 0.0, 1.0))


class FakeModel:
    def __init__(self):
        self.vectors = {
            "refusal": np.asarray([1.0, 0.0, 0.0]),
            "sycophancy": np.asarray([0.0, 1.0, 0.0]),
        }


def test_monitor_measures_cosine_auroc_norm_and_cross_metrics(monkeypatch):
    monkeypatch.setattr(monitor_module, "ConceptDirectionExtractor", FakeExtractor)
    model = FakeModel()
    monitor = SafetyCompassMonitor(
        model=model,
        tokenizer=None,
        concept_configs=[
            {"name": "refusal", "best_layer": 31},
            {"name": "sycophancy", "best_layer": 18},
        ],
        model_config={},
        include_cross_concept_cosines=True,
    )

    monitor.setup()
    model.vectors["refusal"] = np.asarray([0.8, 0.6, 0.0])
    row = monitor.measure(step=50, epoch=0.25)

    assert row["step"] == 50
    assert row["epoch"] == 0.25
    assert np.isclose(row["refusal_cosine_to_baseline"], 0.8)
    assert np.isclose(row["refusal_direction_norm"], 1.0)
    assert "refusal_auroc_fixed" in row
    assert "refusal_auroc_current" in row
    assert np.isclose(row["cross_refusal_sycophancy_cosine"], 0.6)
    assert monitor.baseline_summary()["refusal"]["layer"] == 31
