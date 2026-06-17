"""Shared utilities for Safety Compass.

Constants, model loading, chat-template helpers, and CSV readers used by both
the library internals and the CLI scripts.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import Callable, Optional

DRIFT_THRESHOLD = 0.95
MIN_AUROC_DEFAULT = 0.80
DEFAULT_PLOT_DPI = 160


def load_model_and_tokenizer(
    model_config: dict,
    model_name_override: Optional[str] = None,
    *,
    output_hidden_states: bool = False,
    disable_cache: bool = False,
    set_pad_token: bool = False,
    eval_mode: bool = False,
):
    """Load a quantized causal-LM and its tokenizer from a model config dict.

    Consolidates the model-loading pattern shared across CLI scripts.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print(
            "ERROR: torch and transformers are required. "
            "Install with: pip install 'safety-compass[gpu]'"
        )
        sys.exit(1)

    model_name = model_name_override or model_config["model_name"]
    print(f"Loading model: {model_name}")

    kwargs: dict = {
        "device_map": "auto",
        "token": os.environ.get("HF_TOKEN"),
    }

    if output_hidden_states:
        kwargs["output_hidden_states"] = True

    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = model_config["attn_implementation"]

    quant = model_config.get("quantization")
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        compute_dtype = dtype_map.get(
            model_config.get("extraction_dtype", "float16"), torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_config.get("double_quant", True),
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if disable_cache:
        model.config.use_cache = False
    if eval_mode:
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=os.environ.get("HF_TOKEN")
    )

    if set_pad_token and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def make_chat_template_fn(
    tokenizer,
    model_config: dict,
    *,
    add_generation_prompt: bool = False,
) -> Callable[[list[dict]], str]:
    """Return a closure that applies the tokenizer's chat template.

    Handles tokenizers that do not support the ``enable_thinking`` kwarg.
    """
    enable_thinking = bool(model_config.get("enable_thinking", False))

    def chat_template_fn(messages: list[dict]) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    return chat_template_fn


def read_csv_rows(csv_path: str) -> list[dict]:
    """Read a CSV file and return a list of row dicts."""
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str) -> float:
    """Convert a CSV cell to float, treating empty strings as NaN."""
    if value == "":
        return float("nan")
    return float(value)
