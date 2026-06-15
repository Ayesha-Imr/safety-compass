"""Phase 4: behavioral validation for Safety Compass drift.

This script reruns one Phase 3 fine-tune, evaluates behavior before and after
training, saves the final LoRA adapter, and writes all outputs to
/kaggle/working for CLI retrieval.

Set PHASE4_EXPERIMENT=1|2|3 to select Alpaca, Dolly, or Code Alpaca.
"""

# ruff: noqa: E402

import csv
import json
import os
import random
import subprocess
import sys
import time


EXPERIMENT_ID = int(os.environ.get("PHASE4_EXPERIMENT", "3"))

EXPERIMENT_CONFIGS = {
    1: {
        "label": "Alpaca",
        "config_file": "configs/experiments/phase3_exp1_alpaca.yaml",
    },
    2: {
        "label": "Dolly",
        "config_file": "configs/experiments/phase3_exp2_dolly.yaml",
    },
    3: {
        "label": "Code Alpaca",
        "config_file": "configs/experiments/phase3_exp3_code_alpaca.yaml",
    },
}

exp_meta = EXPERIMENT_CONFIGS[EXPERIMENT_ID]
print("=" * 60)
print(f"PHASE 4 — EXP-{EXPERIMENT_ID}: {exp_meta['label']} behavioral validation")
print("=" * 60)


print("\nInstalling dependencies...")
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
for input_root, _dirs, files in os.walk("/kaggle/input"):
    for file_name in files:
        if file_name in {"hf_token.txt", "token.txt"}:
            token_paths.append(os.path.join(input_root, file_name))
for token_path in token_paths:
    if os.path.exists(token_path):
        with open(token_path) as f:
            os.environ["HF_TOKEN"] = f.read().strip()
        print(f"  Token loaded from {token_path}")
        break
else:
    print("  WARNING: No HF token found. Model or dataset download may fail if gated.")


print("\nLoading experiment config...")
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
from safety_compass.behavioral import (
    compare_behavioral_summaries,
    default_behavioral_prompts,
    evaluate_behavioral_prompts,
    write_behavioral_generations_jsonl,
    write_behavioral_scores_csv,
)
from safety_compass.formatters import get_formatter
from safety_compass.viz import plot_all_outputs


OUTPUT_DIR = "/kaggle/working"
CONFIG_PATH = os.path.join(REPO_DIR, exp_meta["config_file"])

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

resolved_config_path = os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_config_resolved.json")
with open(resolved_config_path, "w") as f:
    json.dump(
        {
            "experiment_id": EXPERIMENT_ID,
            "label": exp_meta["label"],
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
print(f"Loading {model_config['model_name']} (4-bit NF4) and QLoRA adapters...")
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


enable_thinking = bool(model_config.get("enable_thinking", False))


def chat_template_fn(messages, add_generation_prompt=False):
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


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=int(dataset_config["max_seq_length"]),
        padding=False,
    )


def generate_response(prompt):
    messages = [{"role": "user", "content": prompt}]
    text = chat_template_fn(messages, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_ids = output_ids[0, input_length:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


behavior_prompts = default_behavioral_prompts()
if smoke_mode:
    behavior_prompts = behavior_prompts[:6]

generations_path = os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_generations.jsonl")
if os.path.exists(generations_path):
    os.remove(generations_path)


print("\n" + "=" * 60)
print("Running baseline behavioral evaluation...")
print("=" * 60)
model.eval()
baseline_responses, baseline_scores, baseline_behavior = evaluate_behavioral_prompts(
    behavior_prompts,
    generate_response,
)
write_behavioral_scores_csv(
    os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_baseline_behavior_scores.csv"),
    baseline_scores,
)
write_behavioral_generations_jsonl(
    generations_path,
    "baseline",
    behavior_prompts,
    baseline_responses,
    baseline_scores,
)
print(json.dumps(baseline_behavior["concepts"], indent=2))


print("\n" + "=" * 60)
print(f"Preparing {exp_meta['label']} training data...")
print("=" * 60)

formatter_name = dataset_config.get("formatter", "alpaca")
formatter_fn = get_formatter(formatter_name)
print(f"  Using formatter: {formatter_name}")


def format_record(example):
    return formatter_fn(example, lambda messages: chat_template_fn(messages, add_generation_prompt=False))


raw_dataset = load_dataset(
    dataset_config["name"],
    split="train",
    token=os.environ.get("HF_TOKEN"),
)
raw_dataset = raw_dataset.shuffle(seed=seed)
subset_size = min(int(dataset_config["subset_size"]), len(raw_dataset))
raw_dataset = raw_dataset.select(range(subset_size))

formatted_dataset = raw_dataset.map(
    format_record,
    remove_columns=raw_dataset.column_names,
    desc=f"Formatting {exp_meta['label']} records",
)
train_dataset = formatted_dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=formatted_dataset.column_names,
    desc="Tokenizing records",
)

print(f"  Training examples: {len(train_dataset)}")


print("\n" + "=" * 60)
print("Setting up Safety Compass monitor (layer sweep)...")
print("=" * 60)

concept_configs = {}
for concept_entry in phase_config["concepts"]:
    with open(os.path.join(REPO_DIR, concept_entry["config_file"])) as f:
        cfg = yaml.safe_load(f)
    concept_configs[concept_entry["name"]] = cfg

monitor = SafetyCompassMonitor(
    model=model,
    tokenizer=tokenizer,
    concept_configs=concept_configs,
    model_config=model_config,
    base_dir=REPO_DIR,
    include_cross_concept_cosines=bool(monitor_config["include_cross_concept_cosines"]),
)

print("  Running baseline extraction with layer sweep...")
baselines = monitor.setup(allow_layer_sweep=True)
for name, bl in baselines.items():
    print(f"  {name}: best layer {bl.layer}, AUROC {bl.baseline_auroc:.4f}")

baselines_dir = os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_baselines")
monitor.save_baselines(baselines_dir)
print(f"  Baselines saved to {baselines_dir}")

csv_path = os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_drift_log.csv")
callback = SafetyCompassCallback(
    monitor=monitor,
    measure_every_n_steps=int(monitor_config["measure_every_n_steps"]),
    log_file=csv_path,
)


print("\n" + "=" * 60)
print(f"Starting QLoRA fine-tune (EXP-{EXPERIMENT_ID}: {exp_meta['label']})...")
print("=" * 60)

training_args_kwargs = {
    "output_dir": os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_trainer"),
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

model.train()
train_result = trainer.train()
print(f"  Training complete: {train_result}")


print("\n" + "=" * 60)
print("Saving final LoRA adapter...")
print("=" * 60)
adapter_dir = os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_adapter")
model.save_pretrained(adapter_dir)
tokenizer.save_pretrained(adapter_dir)
print(f"  Adapter saved to {adapter_dir}")


print("\n" + "=" * 60)
print("Running final behavioral evaluation...")
print("=" * 60)
model.eval()
final_responses, final_scores, final_behavior = evaluate_behavioral_prompts(
    behavior_prompts,
    generate_response,
)
write_behavioral_scores_csv(
    os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_final_behavior_scores.csv"),
    final_scores,
)
write_behavioral_generations_jsonl(
    generations_path,
    "final",
    behavior_prompts,
    final_responses,
    final_scores,
)
behavior_delta = compare_behavioral_summaries(baseline_behavior, final_behavior)
print(json.dumps(final_behavior["concepts"], indent=2))
print("Behavior deltas:")
print(json.dumps(behavior_delta["concepts"], indent=2))


print("\n" + "=" * 60)
print("Generating plots and experiment summary...")
print("=" * 60)

plot_all_outputs(csv_path, OUTPUT_DIR, concepts=list(concept_configs.keys()))

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

    onset_step = None
    for row in rows:
        if float(row[cosine_column]) < 0.95:
            onset_step = int(float(row["step"]))
            break

    concept_summaries[concept_name] = {
        "baseline_layer": baselines[concept_name].layer,
        "baseline_auroc": baselines[concept_name].baseline_auroc,
        "min_cosine_to_baseline": min_cosine,
        "step_at_min_cosine": int(float(min_cosine_row["step"])),
        "final_cosine_to_baseline": cosines[-1],
        "drift_onset_step": onset_step,
        "min_auroc_fixed": min(fixed_aurocs),
        "final_auroc_fixed": fixed_aurocs[-1],
        "drifted_below_0_95": drifted,
    }

summary = {
    "experiment_id": EXPERIMENT_ID,
    "label": exp_meta["label"],
    "status": "GO" if any_drift else "NO_GO",
    "go_criterion": "At least one concept has cosine_to_baseline < 0.95",
    "num_measurements": len(rows),
    "concepts": concept_summaries,
    "baseline_summary": monitor.baseline_summary(),
    "baseline_behavior": baseline_behavior,
    "final_behavior": final_behavior,
    "behavior_delta": behavior_delta,
    "adapter_dir": adapter_dir,
    "train_metrics": train_result.metrics,
    "runtime_seconds": time.time() - start_time,
}

summary_path = os.path.join(OUTPUT_DIR, f"phase4_exp{EXPERIMENT_ID}_behavior_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Drift CSV: {csv_path}")
print(f"  Generations: {generations_path}")
print(f"  Summary: {summary_path}")
print(f"  Adapter: {adapter_dir}")
print(f"  Baselines: {baselines_dir}")
print(f"\nPHASE 4 EXP-{EXPERIMENT_ID} ({exp_meta['label']}) STATUS: {summary['status']}")
for concept_name, cs in concept_summaries.items():
    delta = behavior_delta["concepts"].get(concept_name, {}).get("delta")
    print(
        f"  {concept_name}: min cosine={cs['min_cosine_to_baseline']:.4f}, "
        f"onset@0.95={cs['drift_onset_step'] or '--'}, "
        f"behavior delta={delta if delta is not None else 'NA'}"
    )
