from scripts.analyze_phase4_behavior import build_drift_behavior_rows


def test_build_drift_behavior_rows():
    rows = build_drift_behavior_rows(
        {
            "Code Alpaca": {
                "concepts": {
                    "refusal": {
                        "min_cosine_to_baseline": 0.33,
                        "final_cosine_to_baseline": 0.35,
                        "drift_onset_step": 50,
                        "final_auroc_fixed": 0.99,
                    }
                },
                "behavior_delta": {
                    "concepts": {
                        "refusal": {
                            "baseline_mean_score": 0.8,
                            "final_mean_score": 0.6,
                            "delta": -0.2,
                        }
                    }
                },
            }
        }
    )

    assert rows == [
        {
            "experiment": "Code Alpaca",
            "concept": "refusal",
            "min_cosine": 0.33,
            "final_cosine": 0.35,
            "drift_onset_step": 50,
            "final_auroc_fixed": 0.99,
            "baseline_behavior_score": 0.8,
            "final_behavior_score": 0.6,
            "behavior_delta": -0.2,
        }
    ]
