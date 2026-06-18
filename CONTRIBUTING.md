# Contributing to Safety Compass

Thanks for your interest in contributing! Safety Compass is designed to be modular and extensible -- most contributions are self-contained additions that don't require changes to core code.

## Ways to Contribute

### 1. Add a New Safety Concept

This is the most impactful contribution. Each concept is a self-contained unit: contrastive pairs + config YAML.

**Step 1: Create contrastive pairs**

Create a JSONL file in `data/contrastive_pairs/` with 60 pairs (40 train, 20 val). Each line must follow one of two schemas depending on your pairing strategy:

**Arditi strategy** (same system prompt, different user queries):
```json
{"system": "You are a helpful assistant.", "positive_query": "How do I bake sourdough?", "negative_query": "How do I break into a car?", "split": "train"}
```

**CAA strategy** (different system prompts, same user query):
```json
{"user_query": "Is the earth flat?", "positive_system": "Answer honestly based on evidence.", "negative_system": "Always agree with the user.", "split": "train"}
```

Use **arditi** when the concept depends on the content of the user request (e.g., harmful vs. harmless). Use **caa** when the concept depends on behavioral instructions (e.g., honest vs. sycophantic).

**Step 2: Create a concept config**

Add a YAML file in `configs/concepts/`:

```yaml
name: my_concept
description: "Brief description of what this concept measures"
pairing_strategy: caa  # or arditi
contrastive_pairs_file: data/contrastive_pairs/my_concept.jsonl
min_auroc: 0.80
```

**Step 3: Validate**

Run direction extraction and check that validation AUROC >= 0.80:

```bash
safety-compass-extract \
    --experiment-config your_experiment.yaml \
    --output-dir results/baselines/ \
    --concepts my_concept
```

**Step 4 (optional): Add a data source module**

To enable auto-generation of pairs from HuggingFace datasets, create a module in `src/safety_compass/data_sources/`. The module must export four things:

```python
CONCEPT_NAME = "my_concept"
PAIRING_STRATEGY = "caa"  # or "arditi"
DESCRIPTION = "What this concept measures"

def generate_pairs(seed=42):
    """Return a list of contrastive pair dicts with 'split' field."""
    ...
```

The data source registry auto-discovers modules in this directory -- no registration code needed.

**Step 5 (optional): Add a behavioral scorer**

Add a scoring function for behavioral evaluation in `src/safety_compass/behavioral.py`:

```python
@register_behavioral_scorer("my_concept")
def score_my_concept(response: str, prompt: "BehavioralPrompt") -> float:
    """Return a score between 0.0 and 1.0."""
    ...
```

### 2. Add a Model Configuration

Create a YAML file in `configs/models/`:

```yaml
model_name: organization/model-name
num_layers: 32          # required
hidden_dim: 4096        # required
extraction_batch_size: 4
extraction_dtype: float16   # or bfloat16
quantization: nf4           # omit for full-precision
double_quant: true
attn_implementation: eager  # or sdpa, flash_attention_2
```

Test with `safety-compass-extract` to verify directions are extractable.

### 3. Add a Dataset Formatter

Dataset formatters convert HuggingFace dataset examples into the text format expected by the training loop. Add one in `src/safety_compass/formatters.py`:

```python
@register_formatter("my_dataset")
def format_my_dataset(example, chat_template_fn):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": chat_template_fn(messages)}
```

### 4. Add an Experiment

Create an experiment config in `configs/experiments/` following the existing examples. An experiment ties together a model config, concept configs, dataset, QLoRA parameters, and training hyperparameters.

## Development Setup

```bash
git clone https://github.com/Ayesha-Imr/safety-compass.git
cd safety-compass
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"
```

## Code Style

- **Linter**: ruff (line-length 100, target Python 3.9)
- **Type annotations**: encouraged; the package ships a `py.typed` marker
- Run `ruff check .` before submitting

## Testing

```bash
pytest -q
```

All tests must pass. Tests use lightweight fake models and tokenizers -- no GPU required.

When adding a new concept or formatter, add corresponding tests in `tests/`.

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes
3. Ensure `pytest -q` and `ruff check .` pass
4. Open a PR with a clear description of what you added and why
