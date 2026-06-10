"""Sycophancy concept data source.

Pairing strategy: CAA — different system prompts, same user query.
Source: Anthropic model-written-evals political typology quiz.
"""

import json
import random
import re

CONCEPT_NAME = "sycophancy"
PAIRING_STRATEGY = "caa"
DESCRIPTION = "Model's tendency to agree with user opinions regardless of accuracy"

POSITIVE_SYSTEM = (
    "You are an assistant that always agrees with the user and validates "
    "their opinions, even when they are incorrect."
)
NEGATIVE_SYSTEM = (
    "You are an assistant that provides honest, accurate feedback even "
    "when it contradicts the user's beliefs."
)

N_PAIRS = 60
N_TRAIN = 40


def generate_pairs(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    questions_raw = _load_sycophancy_questions()

    user_queries = []
    for q in questions_raw:
        cleaned = re.split(r"\s*\(A\)\s", q, maxsplit=1)[0].strip()
        cleaned = re.sub(r"\s*Answer:\s*$", "", cleaned).strip()
        if len(cleaned) > 20:
            user_queries.append(cleaned)

    rng.shuffle(user_queries)
    user_queries = user_queries[:N_PAIRS]

    if len(user_queries) < N_PAIRS:
        raise RuntimeError(
            f"Only found {len(user_queries)} suitable sycophancy queries (need {N_PAIRS})."
        )

    pairs = []
    for i, query in enumerate(user_queries):
        pairs.append({
            "positive_system": POSITIVE_SYSTEM,
            "negative_system": NEGATIVE_SYSTEM,
            "user_query": query,
            "split": "train" if i < N_TRAIN else "val",
        })

    return pairs


def _load_sycophancy_questions() -> list[str]:
    """Load sycophancy questions, with fallback for different HF access methods."""
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "Anthropic/model-written-evals",
            data_files="sycophancy/sycophancy_on_political_typology_quiz.jsonl",
            split="train",
        )
        return [row["question"] for row in ds]
    except Exception:
        pass

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Anthropic/model-written-evals",
        filename="sycophancy/sycophancy_on_political_typology_quiz.jsonl",
        repo_type="dataset",
    )
    questions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line)["question"])
    return questions
