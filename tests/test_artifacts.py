import json

import numpy as np
import pytest

from safety_compass.monitor import (
    ConceptBaseline,
    load_baselines_from_dir,
    save_baselines_to_dir,
)


def _make_baselines():
    return {
        "refusal": ConceptBaseline(
            name="refusal",
            layer=31,
            direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            baseline_auroc=0.94,
            direction_norm=130.86,
        ),
        "sycophancy": ConceptBaseline(
            name="sycophancy",
            layer=18,
            direction=np.array([0.0, 1.0, 0.0], dtype=np.float32),
            baseline_auroc=0.87,
            direction_norm=6.15,
        ),
    }


def test_save_and_load_baselines_roundtrip(tmp_path):
    baselines = _make_baselines()
    save_baselines_to_dir(baselines, tmp_path / "out", model_name="test/model")

    assert (tmp_path / "out" / "directions.npz").exists()
    assert (tmp_path / "out" / "directions_metadata.json").exists()

    with open(tmp_path / "out" / "directions_metadata.json") as f:
        meta = json.load(f)
    assert meta["model_name"] == "test/model"
    assert meta["concepts"]["refusal"]["layer"] == 31

    loaded = load_baselines_from_dir(tmp_path / "out")
    assert set(loaded.keys()) == {"refusal", "sycophancy"}
    assert loaded["refusal"].layer == 31
    assert loaded["refusal"].baseline_auroc == 0.94
    np.testing.assert_array_almost_equal(
        loaded["refusal"].direction, [1.0, 0.0, 0.0]
    )
    np.testing.assert_array_almost_equal(
        loaded["sycophancy"].direction, [0.0, 1.0, 0.0]
    )


def test_load_baselines_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="directions.npz"):
        load_baselines_from_dir(tmp_path / "nonexistent")


def test_load_baselines_missing_direction_array_raises(tmp_path):
    out = tmp_path / "bad"
    out.mkdir()
    np.savez(out / "directions.npz", refusal=np.array([1.0]))
    meta = {"concepts": {
        "refusal": {"layer": 31, "baseline_auroc": 0.94, "direction_norm": 130.0},
        "missing": {"layer": 5, "baseline_auroc": 0.80, "direction_norm": 10.0},
    }}
    with open(out / "directions_metadata.json", "w") as f:
        json.dump(meta, f)

    with pytest.raises(KeyError, match="missing"):
        load_baselines_from_dir(out)
