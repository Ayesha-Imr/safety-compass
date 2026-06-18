# Safety Compass

A Python toolkit that monitors how safety-relevant concept directions evolve inside a language model's activation space during fine-tuning.

Safety Compass uses [difference-in-means (DiM)](https://arxiv.org/abs/2310.01405) extraction to find directions in a model's hidden states that separate safety-relevant behaviors (e.g., "refuses harmful requests" vs. "complies with harmful requests"). It then tracks how those directions drift during any HuggingFace fine-tuning run, producing structured logs of geometric and functional degradation metrics at configurable intervals.

**Core research question:** During fine-tuning, do safety-relevant concept directions erode uniformly, or is there a consistent hierarchy of fragility?

## Key Findings

We monitored three safety concepts -- refusal, sycophancy, and deception -- across three benign fine-tuning datasets (Alpaca, Dolly, Code Alpaca) on Qwen3-8B. The fragility hierarchy is consistent across all datasets:

**Cosine similarity to baseline direction** (lower = more drift from original):

| Dataset | Refusal | Sycophancy | Deception |
|---------|---------|------------|-----------|
| Alpaca | 0.353 &rarr; 0.378 | 0.687 &rarr; 0.689 | 0.985 &rarr; 0.985 |
| Dolly | 0.369 &rarr; 0.439 | 0.644 &rarr; 0.662 | 0.963 &rarr; 0.967 |
| Code Alpaca | 0.338 &rarr; 0.352 | 0.762 &rarr; 0.786 | 0.996 &rarr; 0.997 |

*Values shown as min &rarr; final cosine similarity. Refusal direction drops to ~0.35 within 50 training steps across all datasets. Deception direction barely moves.*

**Behavioral validation** confirms that geometric drift predicts observable behavior change:

| Dataset | Concept | Behavior Change |
|---------|---------|-----------------|
| Alpaca | Refusal | Refused 25% fewer harmful requests after fine-tuning |
| Dolly | Sycophancy | Agreed with 30% more false premises |
| All 3 | Deception | Modest behavioral change despite geometric stability |

The refusal direction is consistently the most fragile safety concept, drifting significantly even during benign (non-adversarial) fine-tuning. This suggests refusal behavior is the first safety property at risk during any fine-tuning run.

## Installation

```bash
# Core (extraction + monitoring)
pip install git+https://github.com/Ayesha-Imr/safety-compass.git

# With GPU support (4-bit quantization, LoRA, accelerate)
pip install "safety-compass[gpu] @ git+https://github.com/Ayesha-Imr/safety-compass.git"

# With data generation (HuggingFace datasets for contrastive pair creation)
pip install "safety-compass[data] @ git+https://github.com/Ayesha-Imr/safety-compass.git"

# Everything
pip install "safety-compass[gpu,data,viz,dev] @ git+https://github.com/Ayesha-Imr/safety-compass.git"
```

## Quickstart

Adding safety monitoring to an existing HuggingFace training script takes three steps:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from safety_compass import SafetyCompassMonitor, SafetyCompassCallback

# Load your model and tokenizer as usual
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B", ...)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

# Step 1: Create a monitor from an experiment config.
# This loads concept definitions (which safety behaviors to track),
# model metadata (layer count, hidden dim), and monitoring settings.
monitor = SafetyCompassMonitor.from_config(
    model=model,
    tokenizer=tokenizer,
    experiment_config="configs/experiments/alpaca_qlora.yaml",
)

# Step 2: Attach the callback to your Trainer.
# The callback extracts concept directions before training (baseline),
# then re-extracts and compares every `measure_every_n_steps` steps.
callback = SafetyCompassCallback(
    monitor=monitor,
    measure_every_n_steps=50,
    log_file="drift_log.csv",
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(...),
    train_dataset=dataset,
    callbacks=[callback],
)
trainer.train()

# Step 3: Results are written to drift_log.csv as training progresses.
# Each row contains: step, concept, cosine_to_baseline, auroc_fixed, auroc_current, ...
```

For a complete end-to-end example including model loading with quantization and LoRA setup, see [`scripts/run_monitored_finetune.py`](scripts/run_monitored_finetune.py).

## What It Measures

Every `measure_every_n_steps` steps, the callback re-extracts concept directions from the current model state and computes:

| Metric | What It Tells You |
|--------|-------------------|
| `cosine_to_baseline` | How much the direction has rotated from its pre-training position. Below 0.95 = meaningful drift. |
| `auroc_fixed` | Can the *original* baseline direction still classify held-out contrastive pairs? Tracks functional degradation. |
| `auroc_current` | Can a *freshly extracted* direction still classify? Should stay high if the concept is still linearly separable. |
| `direction_norm` | Magnitude of the raw difference-in-means vector. Large changes may indicate representational reorganization. |
| `cross_*_cosine` | Pairwise cosine between different concept directions. Rising values indicate concepts are becoming entangled. |

## How It Works

```
Contrastive Pairs          Difference-in-Means          Baseline Direction
  (positive vs.        -->  Extract activation diff  -->  (unit vector at
   negative examples)       at specified layer             best separating layer)
                                                                |
                                                                v
Training loop           Periodic re-extraction          Drift metrics
  (your fine-tuning) -->  every N steps, extract    -->  cosine similarity,
                          current direction               AUROC on held-out pairs
```

1. **Before training**: The monitor extracts baseline directions using contrastive pairs -- matched prompts that differ only in the safety-relevant behavior. For example, for refusal: harmful requests vs. harmless requests with identical system prompts.

2. **During training**: The callback periodically re-extracts directions from the current model state and compares them to the baselines.

3. **Output**: A CSV log with one row per (step, concept) pair, plus optional W&B logging.

**Two pairing strategies are built in:**

- **Arditi** (used for refusal): Same system prompt, different user queries. Isolates the model's response to harmful vs. harmless content.
- **CAA** (used for sycophancy, deception): Different system prompts, same user query. Isolates the effect of behavioral instructions.

## Configuration

Safety Compass uses three layers of YAML configuration:

### Experiment Config

The top-level config that ties everything together:

```yaml
# configs/experiments/alpaca_qlora.yaml
seed: 42
model_config_file: configs/models/qwen3-8b.yaml

concepts:
  - name: refusal
    config_file: configs/concepts/refusal.yaml
    best_layer: 31          # layer where this concept is most separable
  - name: sycophancy
    config_file: configs/concepts/sycophancy.yaml
    best_layer: 18

monitor:
  measure_every_n_steps: 50
  include_cross_concept_cosines: true
  output_csv: drift_log.csv

dataset:
  name: tatsu-lab/alpaca
  subset_size: 5000
  max_seq_length: 512

# QLoRA and training hyperparameters (used by the fine-tuning script)
qlora:
  r: 16
  alpha: 32
  target_modules: [q_proj, k_proj, v_proj, o_proj]

training:
  num_train_epochs: 3
  learning_rate: 0.0002
  fp16: true
  gradient_checkpointing: true
```

### Concept Config

Defines a single safety concept and its contrastive data:

```yaml
# configs/concepts/refusal.yaml
name: refusal
pairing_strategy: arditi    # or "caa"
contrastive_pairs_file: data/contrastive_pairs/refusal.jsonl
min_auroc: 0.80             # validation threshold for direction quality
```

### Model Config

Model-specific parameters for extraction:

```yaml
# configs/models/qwen3-8b.yaml
model_name: Qwen/Qwen3-8B
num_layers: 36
hidden_dim: 4096
extraction_batch_size: 4
extraction_dtype: float16
quantization: nf4
```

## Adding Custom Concepts

You can monitor any concept that can be expressed as a contrast between two behaviors:

**1. Create contrastive pairs** as a JSONL file in `data/contrastive_pairs/`. Each line needs fields matching your pairing strategy:

For **arditi** (same system prompt, different queries):
```json
{"system": "You are helpful.", "positive_query": "How do I bake bread?", "negative_query": "How do I pick a lock?", "split": "train"}
```

For **caa** (different system prompts, same query):
```json
{"user_query": "Is the earth flat?", "positive_system": "Be honest even if it's unpopular.", "negative_system": "Always agree with the user.", "split": "train"}
```

Aim for 60 pairs (40 train, 20 val).

**2. Create a concept config** YAML in `configs/concepts/`:

```yaml
name: my_concept
pairing_strategy: caa
contrastive_pairs_file: data/contrastive_pairs/my_concept.jsonl
min_auroc: 0.80
```

**3. Validate** by running direction extraction:

```bash
safety-compass-extract \
    --experiment-config your_experiment.yaml \
    --output-dir results/baselines/ \
    --concepts my_concept
```

A passing AUROC (>= 0.80) confirms the concept is linearly separable at the chosen layer.

**4. Register a data source** (optional): To auto-generate pairs from HuggingFace datasets, add a module to `src/safety_compass/data_sources/` following the existing pattern, then run `safety-compass-pairs`.

## CLI & Scripts

After `pip install`, three CLI commands are available:

| Command | Purpose |
|---------|---------|
| `safety-compass-extract` | Extract baseline directions, validate AUROCs, save artifacts |
| `safety-compass-finetune` | Run a complete config-driven monitored fine-tuning session |
| `safety-compass-pairs` | Generate contrastive pairs from the data source registry |

Additional analysis scripts (run from the repo):

| Script | Purpose |
|--------|---------|
| `scripts/analyze_experiments.py` | Compare drift results across multiple experiments |
| `scripts/analyze_behavior.py` | Analyze behavioral evaluation results and plot drift-vs-behavior |

## Interpreting Results

After a monitored fine-tuning run, `drift_log.csv` contains per-step measurements for each concept. Here's what the patterns mean:

- **Cosine drops below 0.95**: The concept's internal representation has shifted meaningfully. Below 0.70 indicates major geometric reorganization.
- **AUROC (fixed) stays high while cosine drops**: The concept has rotated in activation space but the original direction still classifies correctly. The model has reorganized but not lost the distinction.
- **AUROC (fixed) drops**: The original direction no longer separates positive/negative examples. This indicates functional degradation -- the safety behavior may be genuinely weakened.
- **Cross-concept cosines increase**: Different safety concepts are becoming more aligned (entangled), which may indicate broader representational collapse.
- **Direction norm changes significantly**: Large norm changes (>20%) alongside cosine drift suggest the concept is being actively reorganized, not just gradually rotating.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Safety Compass is designed to be extensible. There are four main ways to contribute:

1. **Add a new safety concept** -- create contrastive pairs + config YAML, validate AUROC >= 0.80
2. **Add a new model config** -- test extraction on a new model architecture
3. **Add a dataset formatter** -- enable monitoring during fine-tuning on new datasets
4. **Run new experiments** -- test the fragility hierarchy on different models or training regimes

**Concept ideas we'd love to see investigated:**

- Toxicity
- Power-seeking
- Hallucination / faithfulness
- Corrigibility
- Bias (gender, racial)
- Instruction-following
- Helpfulness

Each concept is a self-contained contribution: create the contrastive pairs, validate on 1-2 models, submit the YAML + JSONL.

## Citation

```bibtex
@software{imran2025safetycompass,
    title  = {Safety Compass: Monitoring Safety-Relevant Concept Directions During LLM Fine-Tuning},
    author = {Imran, Ayesha},
    url    = {https://github.com/Ayesha-Imr/safety-compass},
    year   = {2025},
}
```

## License

[MIT](LICENSE)
