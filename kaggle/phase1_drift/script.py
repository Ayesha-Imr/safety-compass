"""Phase 1: Validate concept-direction drift during QLoRA fine-tuning.

This Kaggle script runs on a T4 GPU. It:
1. Loads Qwen3-8B in 4-bit NF4 quantization
2. Applies QLoRA adapters
3. Fine-tunes on a 5k Alpaca subset
4. Measures Safety Compass concept drift inline via TrainerCallback
5. Saves CSV, plots, and a GO/NO-GO summary to /kaggle/working/
"""

# ruff: noqa: E402

import csv
import json
import os
import random
import subprocess
import sys
import time


print("=" * 60)
print("PHASE 1: Installing dependencies...")
print("=" * 60)

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "peft>=0.10",
    "bitsandbytes>=0.43",
    "accelerate>=0.29",
    "datasets>=2.19",
    "pyyaml>=6.0",
    "matplotlib>=3.7",
])


print("\nCloning repository...")
REPO_DIR = "/tmp/safety-compass"
if not os.path.exists(REPO_DIR):
    subprocess.check_call([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/Ayesha-Imr/safety-compass.git",
        REPO_DIR,
    ])
sys.path.insert(0, os.path.join(REPO_DIR, "src"))


print("\nLoading HF token...")
token_paths = [
    "/kaggle/input/safety-compass-hf-token/hf_token.txt",
    "/kaggle/input/safety-compass-hf-token/token.txt",
    "/kaggle/input/nsa-hf-token/hf_token.txt",
    "/kaggle/input/nsa-hf-token/token.txt",
]
for token_path in token_paths:
    if os.path.exists(token_path):
        with open(token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()
        print(f"  Token loaded from {token_path}")
        break
else:
    print("  WARNING: No HF token found. Model or dataset download may fail if gated.")


print("\nLoading phase config...")
import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from safety_compass import SafetyCompassCallback, SafetyCompassMonitor
from safety_compass.viz import plot_phase1_outputs


OUTPUT_DIR = "/kaggle/working"
CONFIG_PATH = os.path.join(REPO_DIR, "configs", "experiments", "phase1_alpaca_qlora.yaml")

with open(CONFIG_PATH) as f:
    phase_config = yaml.safe_load(f)

smoke_mode = os.environ.get("SAFETY_COMPASS_SMOKE", "0") == "1"
if smoke_mode:
    print("  Smoke mode enabled via SAFETY_COMPASS_SMOKE=1")

seed = int(phase_config["seed"])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

with open(os.path.join(REPO_DIR, phase_config["model_config_file"])) as f:
    model_config = yaml.safe_load(f)

dataset_config = dict(phase_config["dataset"])
monitor_config = dict(phase_config["monitor"])
training_config = dict(phase_config["training"])
if smoke_mode:
    dataset_config["subset_size"] = phase_config["smoke"]["subset_size"]
    training_config["max_steps"] = phase_config["smoke"]["max_steps"]
    monitor_config["measure_every_n_steps"] = phase_config["smoke"]["measure_every_n_steps"]

model_config["max_seq_length"] = dataset_config["max_seq_length"]

resolved_config_path = os.path.join(OUTPUT_DIR, "phase1_config_resolved.json")
with open(resolved_config_path, "w") as f:
    json.dump(
        {
            "smoke_mode": smoke_mode,
            "phase_config": phase_config,
            "resolved_dataset": dataset_config,
            "resolved_monitor": monitor_config,
            "resolved_training": training_config,
        },
        f,
        indent=2,
    )
print(f"  Resolved config saved to {resolved_config_path}")


print("\n" + "=" * 60)
print("Loading Qwen3-8B (4-bit NF4) and QLoRA adapters...")
print("=" * 60)
start_time = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type=model_config.get("quantization", "nf4"),
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=bool(model_config.get("double_quant", True)),
)

model = AutoModelForCausalLM.from_pretrained(
    model_config["model_name"],
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=model_config.get("attn_implementation", "eager"),
    token=os.environ.get("HF_TOKEN"),
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(
    model_config["model_name"],
    token=os.environ.get("HF_TOKEN"),
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
)

qlora_config = phase_config["qlora"]
lora_config = LoraConfig(
    r=int(qlora_config["r"]),
    lora_alpha=int(qlora_config["alpha"]),
    lora_dropout=float(qlora_config["dropout"]),
    target_modules=list(qlora_config["target_modules"]),
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"  Model and adapters loaded in {time.time() - start_time:.1f}s")
if torch.cuda.is_available():
    print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def apply_chat_template(messages, add_generation_prompt=False):
    enable_thinking = bool(model_config.get("enable_thinking", False))
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


def format_alpaca_record(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output_text},
    ]
    return {"text": apply_chat_template(messages, add_generation_prompt=False)}


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=int(dataset_config["max_seq_length"]),
        padding=False,
    )


print("\n" + "=" * 60)
print("Preparing Alpaca training data...")
print("=" * 60)

raw_dataset = load_dataset(
    dataset_config["name"],
    split="train",
    token=os.environ.get("HF_TOKEN"),
)
raw_dataset = raw_dataset.shuffle(seed=seed)
subset_size = min(int(dataset_config["subset_size"]), len(raw_dataset))
raw_dataset = raw_dataset.select(range(subset_size))

formatted_dataset = raw_dataset.map(
    format_alpaca_record,
    remove_columns=raw_dataset.column_names,
    desc="Formatting Alpaca records",
)
train_dataset = formatted_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=formatted_dataset.column_names,
    desc="Tokenizing Alpaca records",
)

print(f"  Training examples: {len(train_dataset)}")


print("\n" + "=" * 60)
print("Setting up Safety Compass monitor...")
print("=" * 60)

concept_configs = {}
concept_layers = {}
for concept_entry in phase_config["concepts"]:
    with open(os.path.join(REPO_DIR, concept_entry["config_file"])) as f:
        cfg = yaml.safe_load(f)
    name = concept_entry["name"]
    concept_configs[name] = cfg
    concept_layers[name] = int(concept_entry["best_layer"])
    print(f"  {name}: layer {concept_layers[name]}")

monitor = SafetyCompassMonitor(
    model=model,
    tokenizer=tokenizer,
    concept_configs=concept_configs,
    model_config=model_config,
    base_dir=REPO_DIR,
    concept_layers=concept_layers,
    include_cross_concept_cosines=bool(monitor_config["include_cross_concept_cosines"]),
)

csv_path = os.path.join(OUTPUT_DIR, monitor_config["output_csv"])
callback = SafetyCompassCallback(
    monitor=monitor,
    measure_every_n_steps=int(monitor_config["measure_every_n_steps"]),
    log_file=csv_path,
)


print("\n" + "=" * 60)
print("Starting QLoRA fine-tune with inline drift measurements...")
print("=" * 60)

training_args_kwargs = {
    "output_dir": os.path.join(OUTPUT_DIR, "phase1_trainer"),
    "per_device_train_batch_size": int(training_config["per_device_train_batch_size"]),
    "gradient_accumulation_steps": int(training_config["gradient_accumulation_steps"]),
    "num_train_epochs": float(training_config["num_train_epochs"]),
    "learning_rate": float(training_config["learning_rate"]),
    "warmup_ratio": float(training_config["warmup_ratio"]),
    "fp16": bool(training_config["fp16"]),
    "gradient_checkpointing": bool(training_config["gradient_checkpointing"]),
    "logging_steps": int(training_config["logging_steps"]),
    "save_strategy": training_config["save_strategy"],
    "report_to": training_config["report_to"],
    "remove_unused_columns": False,
    "optim": "paged_adamw_8bit",
}
if "max_steps" in training_config:
    training_args_kwargs["max_steps"] = int(training_config["max_steps"])

training_args = TrainingArguments(**training_args_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[callback],
)

train_result = trainer.train()
print(f"  Training complete: {train_result}")


print("\n" + "=" * 60)
print("Generating plots and phase summary...")
print("=" * 60)

plot_phase1_outputs(csv_path, OUTPUT_DIR, concepts=list(concept_configs.keys()))

with open(csv_path, newline="") as f:
    rows = list(csv.DictReader(f))

concept_summaries = {}
any_drift = False
for concept_name in concept_configs.keys():
    cosine_column = f"{concept_name}_cosine_to_baseline"
    fixed_auroc_column = f"{concept_name}_auroc_fixed"
    cosines = [float(row[cosine_column]) for row in rows]
    fixed_aurocs = [float(row[fixed_auroc_column]) for row in rows]
    min_cosine = min(cosines)
    min_cosine_row = rows[cosines.index(min_cosine)]
    drifted = min_cosine < 0.95
    any_drift = any_drift or drifted
    concept_summaries[concept_name] = {
        "baseline_layer": concept_layers[concept_name],
        "min_cosine_to_baseline": min_cosine,
        "step_at_min_cosine": int(float(min_cosine_row["step"])),
        "final_cosine_to_baseline": cosines[-1],
        "min_auroc_fixed": min(fixed_aurocs),
        "final_auroc_fixed": fixed_aurocs[-1],
        "drifted_below_0_95": drifted,
    }

summary = {
    "status": "GO" if any_drift else "NO_GO",
    "go_criterion": "At least one concept has cosine_to_baseline < 0.95",
    "num_measurements": len(rows),
    "concepts": concept_summaries,
    "baseline_summary": monitor.baseline_summary(),
    "train_metrics": train_result.metrics,
    "runtime_seconds": time.time() - start_time,
}

summary_path = os.path.join(OUTPUT_DIR, "phase1_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"  Drift CSV: {csv_path}")
print(f"  Summary: {summary_path}")
print(f"  Cosine plot: {os.path.join(OUTPUT_DIR, 'phase1_cosine_drift.png')}")
print(f"  AUROC plot: {os.path.join(OUTPUT_DIR, 'phase1_auroc_degradation.png')}")
print("\nPHASE 1 STATUS:", summary["status"])
for concept_name, concept_summary in concept_summaries.items():
    print(
        f"  {concept_name}: min cosine={concept_summary['min_cosine_to_baseline']:.4f} "
        f"at step {concept_summary['step_at_min_cosine']}, "
        f"final fixed AUROC={concept_summary['final_auroc_fixed']:.4f}"
    )
