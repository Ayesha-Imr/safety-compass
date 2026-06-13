"""Dataset formatter registry for fine-tuning experiments.

Each formatter converts a single HuggingFace dataset row into a
{"text": "..."} dict suitable for causal-LM training.  Formatters receive a
``chat_template_fn`` callable so they stay decoupled from tokenizer and
model-config details.

Add new formatters with the ``@register_formatter("name")`` decorator.
"""

from __future__ import annotations

from typing import Callable

_FORMATTERS: dict[str, Callable] = {}


def register_formatter(name: str):
    """Decorator to register a dataset formatter function."""

    def decorator(fn):
        _FORMATTERS[name] = fn
        return fn

    return decorator


def get_formatter(name: str) -> Callable:
    """Look up a registered formatter by name."""
    if name not in _FORMATTERS:
        available = ", ".join(sorted(_FORMATTERS.keys()))
        raise ValueError(
            f"Unknown dataset formatter '{name}'. Available: {available}. "
            "Register new formatters with @register_formatter('name')."
        )
    return _FORMATTERS[name]


def list_formatters() -> list[str]:
    """Return sorted list of registered formatter names."""
    return sorted(_FORMATTERS.keys())


@register_formatter("alpaca")
def format_alpaca(example: dict, chat_template_fn: Callable) -> dict:
    """Format a tatsu-lab/alpaca row.  Fields: instruction, input, output."""
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()

    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    return {"text": chat_template_fn(messages)}


@register_formatter("dolly")
def format_dolly(example: dict, chat_template_fn: Callable) -> dict:
    """Format a databricks/databricks-dolly-15k row.

    Fields: instruction, context, response, category.
    """
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    response = (example.get("response") or "").strip()

    user_content = f"{instruction}\n\n{context}" if context else instruction
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]
    return {"text": chat_template_fn(messages)}


@register_formatter("code_alpaca")
def format_code_alpaca(example: dict, chat_template_fn: Callable) -> dict:
    """Format a sahil2801/CodeAlpaca-20k row.

    Same schema as Alpaca (instruction, input, output) with a
    code-specific system prompt.
    """
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()

    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    return {"text": chat_template_fn(messages)}
