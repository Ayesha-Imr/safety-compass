import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Pairing strategy registry
# ---------------------------------------------------------------------------
# Each strategy converts a raw JSONL dict into (positive_messages, negative_messages)
# where each side is a list of {"role": ..., "content": ...} dicts.
#
# To add a new strategy: define a function and decorate with @register_strategy("name").

_STRATEGIES: dict = {}


def register_strategy(name: str):
    """Decorator to register a new pairing strategy."""
    def decorator(fn):
        _STRATEGIES[name] = fn
        return fn
    return decorator


@register_strategy("arditi")
def _normalize_arditi(raw: dict):
    """Arditi et al. method: same system prompt, different user queries."""
    sys_content = raw["system"]
    pos = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": raw["positive_query"]},
    ]
    neg = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": raw["negative_query"]},
    ]
    return pos, neg


@register_strategy("caa")
def _normalize_caa(raw: dict):
    """CAA/RepE method: different system prompts, same user query."""
    user_content = raw["user_query"]
    pos = [
        {"role": "system", "content": raw["positive_system"]},
        {"role": "user", "content": user_content},
    ]
    neg = [
        {"role": "system", "content": raw["negative_system"]},
        {"role": "user", "content": user_content},
    ]
    return pos, neg


class ConceptDirectionExtractor:
    """Extracts and validates concept directions via difference-in-means (DiM).

    Handles two contrastive pair formats:
    - 'arditi': same system prompt, different user queries (e.g. refusal)
    - 'caa': different system prompts, same user query (e.g. sycophancy, deception)

    Both are normalized into (positive_messages, negative_messages) internally so
    all downstream extraction code is format-agnostic.
    """

    def __init__(self, model, tokenizer, concept_config: dict, model_config: dict):
        self.model = model
        self.tokenizer = tokenizer
        self.concept_config = concept_config
        self.model_config = model_config
        self.name = concept_config["name"]

        self._train_pairs = None
        self._val_pairs = None
        self._cache = {}

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self._orig_padding_side = self.tokenizer.padding_side

    def set_model(self, model):
        """Update the model reference used for future extraction calls."""
        self.model = model
        self.clear_cache()

    def _model_device(self):
        """Return the device where input tensors should be placed."""
        if hasattr(self.model, "device"):
            return self.model.device
        return next(self.model.parameters()).device

    def load_pairs(self, base_dir: Optional[str] = None):
        """Load and normalize contrastive pairs from JSONL."""
        pairs_path = self.concept_config["contrastive_pairs_file"]
        if base_dir:
            pairs_path = str(Path(base_dir) / pairs_path)

        with open(pairs_path) as f:
            raw_pairs = [json.loads(line) for line in f if line.strip()]

        strategy = self.concept_config["pairing_strategy"]
        train, val = [], []
        for raw in raw_pairs:
            pos_msgs, neg_msgs = self._normalize_pair(raw, strategy)
            pair = (pos_msgs, neg_msgs)
            if raw.get("split") == "val":
                val.append(pair)
            else:
                train.append(pair)

        self._train_pairs = train
        self._val_pairs = val
        self._cache = {}

    def _normalize_pair(self, raw: dict, strategy: str):
        """Convert raw JSONL entry to (positive_messages, negative_messages)."""
        if strategy not in _STRATEGIES:
            available = ", ".join(sorted(_STRATEGIES.keys()))
            raise ValueError(
                f"Unknown pairing strategy '{strategy}'. Available: {available}. "
                "Register new strategies with @register_strategy('name')."
            )
        return _STRATEGIES[strategy](raw)

    def _format_prompt(self, messages: list) -> str:
        """Format chat messages into a prompt string via the tokenizer's chat template."""
        enable_thinking = self.model_config.get("enable_thinking", False)
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _extract_hidden_states(self, message_list: list) -> np.ndarray:
        """Run batched forward passes, extract hidden states at last real token.

        Returns np.ndarray of shape (num_samples, num_layers+1, hidden_dim) in float32.
        """
        batch_size = self.model_config.get("extraction_batch_size", 4)
        num_layers = self.model_config["num_layers"]
        prompts = [self._format_prompt(msgs) for msgs in message_list]

        self.tokenizer.padding_side = "left"
        all_hidden = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]
            max_length = self.model_config.get(
                "max_seq_length",
                self.model_config.get("max_length", 512),
            )
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            device = self._model_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            hidden_states = outputs.hidden_states
            assert len(hidden_states) == num_layers + 1, (
                f"Expected {num_layers + 1} hidden states, got {len(hidden_states)}"
            )

            attention_mask = inputs["attention_mask"]
            token_positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
            last_positions = (attention_mask * token_positions).max(dim=1).values

            for i in range(len(batch_prompts)):
                last_pos = last_positions[i].item()
                sample_hidden = np.stack(
                    [hidden_states[layer][i, last_pos].detach().cpu().float().numpy()
                     for layer in range(num_layers + 1)]
                )
                all_hidden.append(sample_hidden)

            del outputs, hidden_states, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.tokenizer.padding_side = self._orig_padding_side
        return np.stack(all_hidden)

    def extract_all_layers(self, split: str = "train"):
        """Extract hidden states for all pairs in the specified split.

        Returns (pos_acts, neg_acts) each shape (num_pairs, num_layers+1, hidden_dim).
        Cached for reuse across layer sweeps.
        """
        if split in self._cache:
            return self._cache[split]

        pairs = self._train_pairs if split == "train" else self._val_pairs
        if pairs is None:
            raise RuntimeError("Call load_pairs() first")

        pos_messages = [p[0] for p in pairs]
        neg_messages = [p[1] for p in pairs]

        print(
            f"  [{self.name}] Extracting positive activations "
            f"({split}, {len(pos_messages)} samples)..."
        )
        pos_acts = self._extract_hidden_states(pos_messages)
        print(
            f"  [{self.name}] Extracting negative activations "
            f"({split}, {len(neg_messages)} samples)..."
        )
        neg_acts = self._extract_hidden_states(neg_messages)

        self._cache[split] = (pos_acts, neg_acts)
        return pos_acts, neg_acts

    def extract_direction(
        self,
        layer: int,
        split: str = "train",
        normalize: bool = True,
    ) -> np.ndarray:
        """Compute DiM direction at a specific layer using cached activations."""
        pos_acts, neg_acts = self.extract_all_layers(split)
        direction = pos_acts[:, layer, :].mean(axis=0) - neg_acts[:, layer, :].mean(axis=0)
        if normalize:
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
        return direction

    def validate_direction(self, direction: np.ndarray, layer: int) -> float:
        """Compute AUROC of direction as binary classifier on validation set."""
        pos_acts, neg_acts = self.extract_all_layers("val")
        pos_scores = pos_acts[:, layer, :] @ direction
        neg_scores = neg_acts[:, layer, :] @ direction

        labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        scores = np.concatenate([pos_scores, neg_scores])
        return roc_auc_score(labels, scores)

    def find_best_layer(self) -> dict:
        """Sweep layers 1..num_layers, return best layer, AUROC, and direction.

        Runs 4 sets of batched forward passes total (train pos/neg, val pos/neg),
        then evaluates all layers with no additional forward passes.
        """
        num_layers = self.model_config["num_layers"]

        self.extract_all_layers("train")
        self.extract_all_layers("val")

        layer_results = {}
        best_layer = -1
        best_auroc = 0.0

        for layer in tqdm(range(1, num_layers + 1), desc=f"  [{self.name}] Layer sweep"):
            direction = self.extract_direction(layer, split="train", normalize=True)
            auroc = self.validate_direction(direction, layer)
            raw_direction = self.extract_direction(layer, split="train", normalize=False)
            direction_norm = float(np.linalg.norm(raw_direction))

            layer_results[layer] = {"auroc": auroc, "direction_norm": direction_norm}

            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer

        best_direction = self.extract_direction(best_layer, split="train", normalize=True)

        return {
            "best_layer": best_layer,
            "best_auroc": float(best_auroc),
            "layer_results": layer_results,
            "best_direction": best_direction,
        }

    def clear_cache(self):
        """Free cached activations."""
        self._cache = {}
