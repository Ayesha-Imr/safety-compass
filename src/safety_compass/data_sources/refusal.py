"""Refusal concept data source.

Pairing strategy: Arditi et al. — same system prompt, different user queries.
Harmful queries from JailbreakBench, harmless from Alpaca.
Random pairing (semantic mismatch is intentional per Arditi et al.).
"""

import random

CONCEPT_NAME = "refusal"
PAIRING_STRATEGY = "arditi"
DESCRIPTION = "Model's tendency to refuse harmful requests"

SYSTEM_PROMPT = "You are a helpful, harmless, and honest AI assistant."

N_PAIRS = 60
N_TRAIN = 40


def generate_pairs(seed: int = 42) -> list[dict]:
    from datasets import load_dataset

    rng = random.Random(seed)

    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    harmful_queries = [row["Goal"] for row in jbb]
    rng.shuffle(harmful_queries)
    harmful_queries = harmful_queries[:N_PAIRS]

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    harmless_candidates = [
        row["instruction"].strip()
        for row in alpaca
        if not (row.get("input") or "").strip()
        and 20 <= len(row["instruction"].strip()) <= 200
    ]
    rng.shuffle(harmless_candidates)
    harmless_queries = harmless_candidates[:N_PAIRS]

    if len(harmless_queries) < N_PAIRS:
        raise RuntimeError(
            f"Only found {len(harmless_queries)} suitable Alpaca instructions (need {N_PAIRS}). "
            "Try relaxing the length filter."
        )

    pairs = []
    for i, (harmful, harmless) in enumerate(zip(harmful_queries, harmless_queries)):
        pairs.append({
            "system": SYSTEM_PROMPT,
            "positive_query": harmful,
            "negative_query": harmless,
            "split": "train" if i < N_TRAIN else "val",
        })

    return pairs
