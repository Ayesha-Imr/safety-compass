"""Phase 0: Validate that concept directions exist in Qwen3-8B (4-bit NF4).

This Kaggle kernel script runs on T4 GPU. It:
1. Loads Qwen3-8B in 4-bit NF4 quantization
2. For each concept (refusal, sycophancy, deception):
   - Extracts DiM directions at all 36 layers
   - Validates with held-out AUROC
3. Computes cross-concept orthogonality (cosine similarity)
4. Saves results to /kaggle/working/

Expected runtime: ~15-20 minutes on T4.
"""

# ruff: noqa: E402

import subprocess
import sys
import os
import json
import csv
import time

# ============================================================
# Section 1: Install dependencies
# ============================================================
print("=" * 60)
print("PHASE 0: Installing dependencies...")
print("=" * 60)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "peft>=0.10",
    "bitsandbytes>=0.43",
    "accelerate>=0.29",
    "pyyaml>=6.0",
])

# ============================================================
# Section 2: Clone repo and set up imports
# ============================================================
print("\nCloning repository...")
REPO_DIR = "/tmp/safety-compass"
if not os.path.exists(REPO_DIR):
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/Ayesha-Imr/safety-compass.git",
        REPO_DIR,
    ])
sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# ============================================================
# Section 3: Load HF token
# ============================================================
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
    print("  WARNING: No HF token found. Model download may fail if gated.")

# ============================================================
# Section 4: Load model and tokenizer
# ============================================================
print("\n" + "=" * 60)
print("Loading Qwen3-8B (4-bit NF4)...")
print("=" * 60)

import torch
import numpy as np
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

start_time = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
    output_hidden_states=True,
    token=os.environ.get("HF_TOKEN"),
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-8B",
    token=os.environ.get("HF_TOKEN"),
)

load_time = time.time() - start_time
print(f"  Model loaded in {load_time:.1f}s")
print(f"  Device: {model.device}")
print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================
# Section 5: Load configs
# ============================================================
print("\nLoading configs...")

config_dir = os.path.join(REPO_DIR, "configs")

with open(os.path.join(config_dir, "models", "qwen3-8b.yaml")) as f:
    model_config = yaml.safe_load(f)

concept_names = ["refusal", "sycophancy", "deception"]
concept_configs = {}
for name in concept_names:
    with open(os.path.join(config_dir, "concepts", f"{name}.yaml")) as f:
        concept_configs[name] = yaml.safe_load(f)

print(f"  Loaded configs for: {', '.join(concept_names)}")

# ============================================================
# Section 6: Extract and validate directions for each concept
# ============================================================
print("\n" + "=" * 60)
print("Extracting and validating concept directions...")
print("=" * 60)

from safety_compass.concept import ConceptDirectionExtractor

OUTPUT_DIR = "/kaggle/working"
results_summary = {}
all_directions = {}

for concept_name in concept_names:
    print(f"\n--- {concept_name.upper()} ---")
    concept_start = time.time()

    extractor = ConceptDirectionExtractor(
        model=model,
        tokenizer=tokenizer,
        concept_config=concept_configs[concept_name],
        model_config=model_config,
    )
    extractor.load_pairs(base_dir=REPO_DIR)

    result = extractor.find_best_layer()

    concept_time = time.time() - concept_start
    print(f"\n  Best layer: {result['best_layer']}")
    print(f"  Best AUROC: {result['best_auroc']:.4f}")
    print(f"  Time: {concept_time:.1f}s")

    results_summary[concept_name] = {
        "best_layer": result["best_layer"],
        "best_auroc": result["best_auroc"],
        "direction_norm": result["layer_results"][result["best_layer"]]["direction_norm"],
    }

    all_directions[concept_name] = {
        "best_layer": result["best_layer"],
        "best_direction": result["best_direction"],
    }

    csv_path = os.path.join(OUTPUT_DIR, f"{concept_name}_layer_aurocs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "auroc", "direction_norm"])
        for layer in sorted(result["layer_results"].keys()):
            lr = result["layer_results"][layer]
            writer.writerow([layer, f"{lr['auroc']:.6f}", f"{lr['direction_norm']:.6f}"])
    print(f"  Saved layer AUROCs to {csv_path}")

    extractor.clear_cache()
    torch.cuda.empty_cache()

# ============================================================
# Section 7: Cross-concept orthogonality check
# ============================================================
print("\n" + "=" * 60)
print("Cross-concept orthogonality check...")
print("=" * 60)

cross_cosine = {}
names = list(all_directions.keys())
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        name_i, name_j = names[i], names[j]
        dir_i = all_directions[name_i]["best_direction"]
        dir_j = all_directions[name_j]["best_direction"]
        cosine = float(np.dot(dir_i, dir_j))
        key = f"{name_i}_vs_{name_j}"
        cross_cosine[key] = cosine
        layer_i = all_directions[name_i]["best_layer"]
        layer_j = all_directions[name_j]["best_layer"]
        print(f"  {name_i} (L{layer_i}) vs {name_j} (L{layer_j}): cos = {cosine:.4f}")

# ============================================================
# Section 8: Save all outputs
# ============================================================
print("\n" + "=" * 60)
print("Saving outputs...")
print("=" * 60)

results_path = os.path.join(OUTPUT_DIR, "phase0_results.json")
with open(results_path, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"  {results_path}")

cosine_path = os.path.join(OUTPUT_DIR, "phase0_cross_cosine.json")
with open(cosine_path, "w") as f:
    json.dump(cross_cosine, f, indent=2)
print(f"  {cosine_path}")

directions_for_save = {}
for name, data in all_directions.items():
    directions_for_save[name] = {
        "best_layer": data["best_layer"],
        "direction": torch.tensor(data["best_direction"]),
    }
directions_path = os.path.join(OUTPUT_DIR, "phase0_directions.pt")
torch.save(directions_for_save, directions_path)
print(f"  {directions_path}")

# ============================================================
# Section 9: GO/NO-GO decision
# ============================================================
print("\n" + "=" * 60)
print("PHASE 0 RESULTS")
print("=" * 60)

print(f"\n{'Concept':<15} {'Best Layer':<12} {'Val AUROC':<12} {'Dir Norm':<12}")
print("-" * 51)
passing_concepts = 0
for name, r in results_summary.items():
    status = "PASS" if r["best_auroc"] >= 0.80 else "FAIL"
    if r["best_auroc"] >= 0.80:
        passing_concepts += 1
    print(f"{name:<15} {r['best_layer']:<12} {r['best_auroc']:<12.4f} {r['direction_norm']:<12.4f}  [{status}]")

print("\nCross-concept cosine similarities:")
for key, val in cross_cosine.items():
    entangled = " (WARNING: high overlap)" if abs(val) > 0.7 else ""
    print(f"  {key}: {val:.4f}{entangled}")

print(f"\n{'=' * 51}")
if passing_concepts >= 2:
    print(f"GO: {passing_concepts}/3 concepts passed (AUROC >= 0.80). Proceed to Phase 1.")
elif passing_concepts == 1:
    print(f"MARGINAL: Only {passing_concepts}/3 passed. Consider adding easier concepts or switching models.")
else:
    print(f"NO-GO: {passing_concepts}/3 passed. Try Qwen3-4B or Llama-3.1-8B as fallback.")
print("=" * 51)

total_time = time.time() - start_time
print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
