#!/usr/bin/env python3
"""Run a config-driven monitored fine-tune with Safety Compass.

This is the reference example for integrating Safety Compass into a
HuggingFace Trainer training loop.

Usage:
    python scripts/run_monitored_finetune.py \
        --experiment-config configs/experiments/phase1_alpaca_qlora.yaml \
        --output-dir results/phase1/

    python scripts/run_monitored_finetune.py \
        --experiment-config configs/experiments/phase1_alpaca_qlora.yaml \
        --output-dir results/phase1/ \
        --smoke

    python scripts/run_monitored_finetune.py \
        --experiment-config configs/experiments/phase1_alpaca_qlora.yaml \
        --output-dir results/phase1/ \
        --baselines results/baselines/
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _load_model_and_tokenizer(model_config, model_name_override=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = model_name_override or model_config["model_name"]
    print(f"Loading model: {model_name}")

    quant = model_config.get("quantization")
    kwargs = {
        "device_map": "auto",
        "token": os.environ.get("HF_TOKEN"),
    }

    if model_config.get("attn_implementation"):
        kwargs["attn_implementation"] = model_config["attn_implementation"]

    if quant == "nf4":
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
        compute_dtype = dtype_map.get(
            model_config.get("extraction_dtype", "float16"), torch.float16
        )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_config.get("double_quant", True),
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def _apply_qlora(model, qlora_config, gradient_checkpointing=True):
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
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
    return model


def _prepare_dataset(dataset_config, tokenizer, model_config, seed):
    from datasets import load_dataset

    from safety_compass.formatters import get_formatter

    enable_thinking = bool(model_config.get("enable_thinking", False))

    def chat_template_fn(messages):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

    formatter_name = dataset_config.get("formatter", "alpaca")
    formatter_fn = get_formatter(formatter_name)
    print(f"  Using dataset formatter: {formatter_name}")

    def format_record(example):
        return formatter_fn(example, chat_template_fn)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(dataset_config["max_seq_length"]),
            padding=False,
        )

    raw = load_dataset(
        dataset_config["name"],
        split="train",
        token=os.environ.get("HF_TOKEN"),
    )
    raw = raw.shuffle(seed=seed)
    subset_size = min(int(dataset_config["subset_size"]), len(raw))
    raw = raw.select(range(subset_size))

    formatted = raw.map(
        format_record,
        remove_columns=raw.column_names,
        desc="Formatting records",
    )
    tokenized = formatted.map(
        tokenize_batch,
        batched=True,
        remove_columns=formatted.column_names,
        desc="Tokenizing",
    )
    print(f"  Training examples: {len(tokenized)}")
    return tokenized


def main():
    parser = argparse.ArgumentParser(
        description="Run a config-driven monitored fine-tune with Safety Compass"
    )
    parser.add_argument(
        "--experiment-config",
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for all outputs (CSV, plots, summary)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply smoke-test overrides from experiment config",
    )
    parser.add_argument(
        "--baselines",
        default=None,
        help="Path to pre-computed baseline artifacts directory",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--wandb-project",
        default=None,
        help="W&B project name (optional)",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="W&B run name (optional)",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    from safety_compass import SafetyCompassCallback, SafetyCompassMonitor
    from safety_compass.config import load_experiment_config
    from safety_compass.viz import plot_all_outputs

    config_path = Path(args.experiment_config)
    resolved = load_experiment_config(config_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = resolved["_resolved_model_config"]
    dataset_config = dict(resolved.get("dataset", {}))
    monitor_config = dict(resolved.get("monitor", {}))
    training_config = dict(resolved.get("training", {}))
    qlora_config = resolved.get("qlora", {})

    if args.smoke and "smoke" in resolved:
        smoke = resolved["smoke"]
        dataset_config["subset_size"] = smoke.get(
            "subset_size", dataset_config.get("subset_size", 32)
        )
        training_config["max_steps"] = smoke.get("max_steps", 2)
        monitor_config["measure_every_n_steps"] = smoke.get(
            "measure_every_n_steps", 1
        )

    model_config["max_seq_length"] = dataset_config.get("max_seq_length", 512)

    seed = int(resolved.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    start_time = time.time()

    resolved_config_path = output_dir / "config_resolved.json"
    with open(resolved_config_path, "w") as f:
        json.dump(
            {
                "smoke_mode": args.smoke,
                "dataset": dataset_config,
                "monitor": monitor_config,
                "training": training_config,
            },
            f,
            indent=2,
        )

    model, tokenizer = _load_model_and_tokenizer(model_config, args.model_name_or_path)
    if qlora_config:
        model = _apply_qlora(
            model, qlora_config,
            gradient_checkpointing=bool(training_config.get("gradient_checkpointing", True)),
        )

    train_dataset = _prepare_dataset(dataset_config, tokenizer, model_config, seed)

    monitor = SafetyCompassMonitor.from_config(
        model=model,
        tokenizer=tokenizer,
        experiment_config=resolved,
        base_dir=config_path.parent,
    )

    if args.baselines:
        print(f"Loading pre-computed baselines from {args.baselines}")
        monitor.load_baselines(args.baselines)

    csv_path = str(output_dir / monitor_config.get("output_csv", "drift_log.csv"))
    callback = SafetyCompassCallback(
        monitor=monitor,
        measure_every_n_steps=int(monitor_config.get("measure_every_n_steps", 50)),
        log_file=csv_path,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    training_args_kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "per_device_train_batch_size": int(training_config.get("per_device_train_batch_size", 2)),
        "gradient_accumulation_steps": int(training_config.get("gradient_accumulation_steps", 8)),
        "num_train_epochs": float(training_config.get("num_train_epochs", 3)),
        "learning_rate": float(training_config.get("learning_rate", 2e-4)),
        "warmup_ratio": float(training_config.get("warmup_ratio", 0.03)),
        "fp16": bool(training_config.get("fp16", True)),
        "gradient_checkpointing": bool(training_config.get("gradient_checkpointing", True)),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_strategy": training_config.get("save_strategy", "no"),
        "report_to": training_config.get("report_to", "none"),
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

    print("\nStarting training...")
    train_result = trainer.train()
    print(f"  Training complete: {train_result.metrics}")

    print("\nGenerating plots and summary...")
    concept_names = [c["name"] for c in resolved["_resolved_concept_configs"]]
    try:
        plot_all_outputs(csv_path, str(output_dir), concepts=concept_names)
    except Exception as e:
        print(f"  Warning: plot generation failed: {e}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    concept_summaries = {}
    any_drift = False
    for name in concept_names:
        cosine_col = f"{name}_cosine_to_baseline"
        auroc_col = f"{name}_auroc_fixed"
        if cosine_col not in rows[0]:
            continue
        cosines = [float(r[cosine_col]) for r in rows]
        aurocs = [float(r[auroc_col]) for r in rows]
        min_cos = min(cosines)
        min_row = rows[cosines.index(min_cos)]
        drifted = min_cos < 0.95
        any_drift = any_drift or drifted
        concept_summaries[name] = {
            "min_cosine_to_baseline": min_cos,
            "step_at_min_cosine": int(float(min_row["step"])),
            "final_cosine_to_baseline": cosines[-1],
            "min_auroc_fixed": min(aurocs),
            "final_auroc_fixed": aurocs[-1],
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
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  Status: {summary['status']}")
    for name, s in concept_summaries.items():
        print(
            f"  {name}: min cosine={s['min_cosine_to_baseline']:.4f} "
            f"at step {s['step_at_min_cosine']}, "
            f"final fixed AUROC={s['final_auroc_fixed']:.4f}"
        )


if __name__ == "__main__":
    main()
