# Safety Compass

Monitor how safety-relevant concept directions evolve in a model's activation space during HuggingFace fine-tuning. Uses difference-in-means (DiM) to extract concept directions, then tracks cosine drift and AUROC degradation inline via a `TrainerCallback`.

**Core research question:** During fine-tuning, do safety-relevant concept directions erode uniformly, or is there a consistent hierarchy of fragility?

## Installation

```bash
# Core (extraction + monitoring)
pip install git+https://github.com/Ayesha-Imr/safety-compass.git

# With GPU support (quantization, LoRA, accelerate)
pip install "safety-compass[gpu] @ git+https://github.com/Ayesha-Imr/safety-compass.git"

# With data generation (datasets, huggingface-hub)
pip install "safety-compass[data] @ git+https://github.com/Ayesha-Imr/safety-compass.git"

# Everything
pip install "safety-compass[gpu,data,viz,dev] @ git+https://github.com/Ayesha-Imr/safety-compass.git"
```

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from safety_compass import SafetyCompassMonitor, SafetyCompassCallback

# 1. Load your model and tokenizer (however you normally do it)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", ...)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# 2. Create monitor from experiment config
monitor = SafetyCompassMonitor.from_config(
    model=model,
    tokenizer=tokenizer,
    experiment_config="configs/experiments/phase1_alpaca_qlora.yaml",
)

# 3. Attach callback to your Trainer
callback = SafetyCompassCallback(
    monitor=monitor,
    measure_every_n_steps=50,
    log_file="drift_log.csv",
)

trainer = Trainer(model=model, ..., callbacks=[callback])
trainer.train()
```

For a complete end-to-end example, see [`scripts/run_monitored_finetune.py`](scripts/run_monitored_finetune.py).

## What It Measures

Every `measure_every_n_steps` steps, the callback extracts current concept directions and computes:

| Metric | Description |
|--------|-------------|
| `cosine_to_baseline` | Cosine similarity between current and baseline direction. Values below 0.95 indicate meaningful drift. |
| `auroc_fixed` | AUROC of the original baseline direction on held-out validation pairs. Tracks functional degradation. |
| `auroc_current` | AUROC of the re-extracted current direction. Should stay high if the concept is still separable. |
| `direction_norm` | Magnitude of the unnormalized difference-in-means vector. |
| `cross_*_cosine` | Pairwise cosine between current directions of different concepts. Monitors entanglement. |

## Configuration

Safety Compass uses three layers of YAML configuration:

### Experiment Config

The top-level config that ties everything together. See [`configs/experiments/phase1_alpaca_qlora.yaml`](configs/experiments/phase1_alpaca_qlora.yaml).

```yaml
seed: 42
model_config_file: configs/models/qwen3-8b.yaml

concepts:
  - name: refusal
    config_file: configs/concepts/refusal.yaml
    best_layer: 31          # from Phase 0 validation
  - name: sycophancy
    config_file: configs/concepts/sycophancy.yaml
    best_layer: 18

monitor:
  measure_every_n_steps: 50
  include_cross_concept_cosines: true
  output_csv: drift_log.csv
```

### Concept Config

Defines a single safety concept. See [`configs/concepts/`](configs/concepts/).

```yaml
name: refusal
pairing_strategy: arditi    # or "caa"
contrastive_pairs_file: data/contrastive_pairs/refusal.jsonl
min_auroc: 0.80
```

Two pairing strategies are built in:
- **arditi**: Same system prompt, different user queries (e.g., harmful vs harmless)
- **caa**: Different system prompts, same user query (e.g., sycophantic vs honest)

### Model Config

Model-specific parameters. See [`configs/models/`](configs/models/).

```yaml
model_name: Qwen/Qwen3-8B
num_layers: 36
hidden_dim: 4096
extraction_batch_size: 4
extraction_dtype: float16
quantization: nf4
```

## Adding Custom Concepts

1. **Create contrastive pairs** as a JSONL file in `data/contrastive_pairs/`. Each line needs fields matching your chosen pairing strategy:
   - `arditi`: `system`, `positive_query`, `negative_query`, `split` (train/val)
   - `caa`: `user_query`, `positive_system`, `negative_system`, `split` (train/val)

   Aim for 60 pairs (40 train, 20 val).

2. **Create a concept config** YAML:
   ```yaml
   name: my_concept
   pairing_strategy: caa
   contrastive_pairs_file: data/contrastive_pairs/my_concept.jsonl
   min_auroc: 0.80
   ```

3. **Validate** by running direction extraction:
   ```bash
   python scripts/extract_directions.py \
       --experiment-config your_experiment.yaml \
       --output-dir results/baselines/ \
       --concepts my_concept
   ```

4. **Register a data source** (optional): To auto-generate pairs, add a module to `src/safety_compass/data_sources/` following the existing pattern, then run `python scripts/prepare_contrastive_pairs.py`.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/extract_directions.py` | Extract directions, validate AUROCs, save baseline artifacts |
| `scripts/run_monitored_finetune.py` | End-to-end monitored fine-tuning example |
| `scripts/prepare_contrastive_pairs.py` | Generate contrastive pairs from data source registry |

## Output Interpretation

- **Cosine to baseline below 0.95**: The concept's internal representation has shifted meaningfully from its pre-training location. Lower values indicate more dramatic repositioning.
- **Fixed-direction AUROC dropping**: The original direction no longer separates positive/negative examples well. This indicates functional degradation, not just geometric rotation.
- **Cosine drifts but AUROC stays high**: The concept has moved geometrically but the original direction still classifies well. The concept is reorganizing but not disappearing.
- **Cross-concept cosines increasing**: Different safety concepts are becoming more aligned (entangled) during fine-tuning.

## Phase 0/1 Results

Validation on Qwen3-8B with Alpaca QLoRA fine-tuning (5k examples, 3 epochs):

| Concept | Phase 0 AUROC | Phase 1 Final Cosine | Phase 1 Final Fixed AUROC |
|---------|--------------|---------------------|--------------------------|
| Refusal | 0.94 | 0.6171 | 0.99 |
| Sycophancy | 0.87 | 0.3678 | 1.0 |
| Deception | 0.99 | 0.5249 | 1.0 |

All three concepts drifted significantly during benign fine-tuning, with sycophancy showing the largest geometric drift and refusal being the only concept with (minor) AUROC degradation.

Detailed results and plots are in the [docs repository](safety-compass-docs/results/phase1/).

## Development

```bash
git clone https://github.com/Ayesha-Imr/safety-compass.git
cd safety-compass
pip install -e ".[dev,viz]"
pytest -q
ruff check .
```

## License

MIT
