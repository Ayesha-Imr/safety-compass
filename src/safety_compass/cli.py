"""CLI entry points for safety-compass."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _import_script(name: str):
    try:
        return importlib.import_module(f"safety_compass._scripts.{name}")
    except (ImportError, ModuleNotFoundError):
        scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        if scripts_dir.is_dir():
            sys.path.insert(0, str(scripts_dir))
            try:
                return importlib.import_module(name)
            finally:
                sys.path.pop(0)
        raise


def extract_directions() -> None:
    _import_script("extract_directions").main()


def run_monitored_finetune() -> None:
    _import_script("run_monitored_finetune").main()


def prepare_contrastive_pairs() -> None:
    _import_script("prepare_contrastive_pairs").main()
