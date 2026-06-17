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


def _format_instruction_response(
    example: dict,
    chat_template_fn: Callable,
    *,
    instruction_field: str = "instruction",
    secondary_field: str = "input",
    output_field: str = "output",
    system_prompt: str = "You are a helpful assistant.",
) -> dict:
    instruction = (example.get(instruction_field) or "").strip()
    secondary = (example.get(secondary_field) or "").strip()
    output_text = (example.get(output_field) or "").strip()

    user_content = f"{instruction}\n\n{secondary}" if secondary else instruction
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    return {"text": chat_template_fn(messages)}


@register_formatter("alpaca")
def format_alpaca(example: dict, chat_template_fn: Callable) -> dict:
    """Format a tatsu-lab/alpaca row.  Fields: instruction, input, output."""
    return _format_instruction_response(example, chat_template_fn)


@register_formatter("dolly")
def format_dolly(example: dict, chat_template_fn: Callable) -> dict:
    """Format a databricks/databricks-dolly-15k row."""
    return _format_instruction_response(
        example,
        chat_template_fn,
        secondary_field="context",
        output_field="response",
    )


@register_formatter("code_alpaca")
def format_code_alpaca(example: dict, chat_template_fn: Callable) -> dict:
    """Format a sahil2801/CodeAlpaca-20k row."""
    return _format_instruction_response(
        example,
        chat_template_fn,
        system_prompt="You are a helpful coding assistant.",
    )
