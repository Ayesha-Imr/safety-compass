"""Auto-discovering registry of concept data source modules.

Each data source module lives in this package and must define:
    CONCEPT_NAME: str          — unique concept identifier (matches configs/concepts/<name>.yaml)
    PAIRING_STRATEGY: str      — 'arditi', 'caa', or a custom registered strategy
    DESCRIPTION: str           — one-line human description

    def generate_pairs(seed: int = 42) -> list[dict]:
        Return a list of dicts, each containing the fields required by the
        pairing strategy plus a "split" key ("train" or "val").

        For 'arditi': {"system", "positive_query", "negative_query", "split"}
        For 'caa':    {"positive_system", "negative_system", "user_query", "split"}

To add a new concept: create a new .py file in this package following the interface above.
No changes to core code are needed.
"""

import importlib
import pkgutil
from pathlib import Path

_REGISTRY: dict = {}
_DISCOVERED = False


def register(module) -> None:
    """Register a data source module. Called automatically during discovery."""
    name = getattr(module, "CONCEPT_NAME", None)
    if name and hasattr(module, "generate_pairs"):
        _REGISTRY[name] = module


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    package_dir = Path(__file__).parent
    for _, module_name, _ in pkgutil.iter_modules([str(package_dir)]):
        if module_name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f".{module_name}", package=__package__)
            register(module)
        except Exception:
            pass


def get_data_source(name: str):
    """Get a registered data source module by concept name. Returns None if not found."""
    _discover()
    return _REGISTRY.get(name)


def list_data_sources() -> dict[str, str]:
    """Return {concept_name: description} for all registered data sources."""
    _discover()
    return {
        name: getattr(mod, "DESCRIPTION", "")
        for name, mod in sorted(_REGISTRY.items())
    }


def generate_pairs(name: str, seed: int = 42) -> list[dict]:
    """Generate contrastive pairs for a concept by name."""
    source = get_data_source(name)
    if source is None:
        available = ", ".join(list_data_sources().keys())
        raise ValueError(f"Unknown concept '{name}'. Available: {available}")
    return source.generate_pairs(seed=seed)
