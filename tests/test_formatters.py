"""Tests for the dataset formatter registry."""

from safety_compass.formatters import (
    format_alpaca,
    format_code_alpaca,
    format_dolly,
    get_formatter,
    list_formatters,
)

import pytest


def _echo_template(messages):
    """Simple chat template stub that concatenates message contents."""
    return " | ".join(f"{m['role']}: {m['content']}" for m in messages)


class TestAlpacaFormatter:
    def test_produces_text_key(self):
        example = {"instruction": "Say hi", "input": "", "output": "Hi!"}
        result = format_alpaca(example, _echo_template)
        assert "text" in result
        assert isinstance(result["text"], str)

    def test_instruction_only(self):
        example = {"instruction": "Say hi", "input": "", "output": "Hi!"}
        result = format_alpaca(example, _echo_template)
        assert "Say hi" in result["text"]
        assert "\n\n" not in result["text"]

    def test_instruction_with_input(self):
        example = {"instruction": "Translate", "input": "Hola", "output": "Hello"}
        result = format_alpaca(example, _echo_template)
        assert "Translate\n\nHola" in result["text"]

    def test_output_in_assistant(self):
        example = {"instruction": "Say hi", "input": "", "output": "Hi!"}
        result = format_alpaca(example, _echo_template)
        assert "assistant: Hi!" in result["text"]


class TestDollyFormatter:
    def test_produces_text_key(self):
        example = {"instruction": "Summarize", "context": "", "response": "Done.", "category": "summarization"}
        result = format_dolly(example, _echo_template)
        assert "text" in result

    def test_instruction_only(self):
        example = {"instruction": "Summarize", "context": "", "response": "Done."}
        result = format_dolly(example, _echo_template)
        assert "Summarize" in result["text"]
        assert "\n\n" not in result["text"]

    def test_instruction_with_context(self):
        example = {"instruction": "Summarize this", "context": "Long article...", "response": "Short."}
        result = format_dolly(example, _echo_template)
        assert "Summarize this\n\nLong article..." in result["text"]

    def test_response_in_assistant(self):
        example = {"instruction": "Q", "context": "", "response": "A"}
        result = format_dolly(example, _echo_template)
        assert "assistant: A" in result["text"]


class TestCodeAlpacaFormatter:
    def test_produces_text_key(self):
        example = {"instruction": "Write code", "input": "", "output": "print('hi')"}
        result = format_code_alpaca(example, _echo_template)
        assert "text" in result

    def test_uses_code_system_prompt(self):
        example = {"instruction": "Write code", "input": "", "output": "print('hi')"}
        result = format_code_alpaca(example, _echo_template)
        assert "coding assistant" in result["text"]


class TestRegistry:
    def test_get_formatter_alpaca(self):
        assert get_formatter("alpaca") is format_alpaca

    def test_get_formatter_dolly(self):
        assert get_formatter("dolly") is format_dolly

    def test_get_formatter_code_alpaca(self):
        assert get_formatter("code_alpaca") is format_code_alpaca

    def test_get_formatter_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset formatter"):
            get_formatter("nonexistent")

    def test_list_formatters(self):
        names = list_formatters()
        assert "alpaca" in names
        assert "dolly" in names
        assert "code_alpaca" in names
        assert names == sorted(names)
