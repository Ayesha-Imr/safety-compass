"""Layer sweep: find best layers for refusal/sycophancy/deception on Llama 3 8B.

Installs safety-compass from PyPI, loads Meta-Llama-3-8B-Instruct with 4-bit
quantization, downloads contrastive pairs from the repo, and runs find_best_layer()
for all three concepts.

Push with: kaggle kernels push -p <dir> --accelerator NvidiaTeslaT4
Expected runtime: ~30 minutes on T4.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

start_time = time.time()

# ============================================================
# 1. Install from PyPI
# ============================================================
print("=" * 60)
print("LAYER SWEEP: Installing safety-compass from PyPI...")
print("=" * 60)

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "safety-compass[gpu,data]",
])

# ============================================================
# 2. Read HF token
# ============================================================
print("\nSearching for HF token...")
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            print(f"  Found: {os.path.join(root, f)}")
else:
    print("  /kaggle/input does not exist")

hf_token = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if "token" in f.lower() or "hf" in f.lower():
            path = os.path.join(root, f)
            hf_token = open(path).read().strip()
            print(f"  HF token loaded from {path}")
            break
    if hf_token:
        break

if not hf_token:
    raise RuntimeError("No HF token found. Attach ayeshaimr/nsa-hf-token dataset.")

os.environ["HF_TOKEN"] = hf_token

# ============================================================
# 3. Download contrastive pairs from GitHub
# ============================================================
WORK_DIR = "/kaggle/working/layer_sweep"
os.makedirs(WORK_DIR, exist_ok=True)
PAIRS_DIR = os.path.join(WORK_DIR, "pairs")
os.makedirs(PAIRS_DIR, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/Ayesha-Imr/safety-compass/main/data/contrastive_pairs"
for name in ["refusal", "sycophancy", "deception"]:
    url = f"{BASE_URL}/{name}.jsonl"
    dest = os.path.join(PAIRS_DIR, f"{name}.jsonl")
    print(f"  Downloading {name}.jsonl...")
    urllib.request.urlretrieve(url, dest)
    with open(dest) as f:
        n = sum(1 for _ in f)
    print(f"    {n} pairs downloaded")

# ============================================================
# 4. Load model
# ============================================================
print("\n" + "=" * 60)
print("Loading Meta-Llama-3-8B-Instruct (4-bit NF4)...")
print("=" * 60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
NUM_LAYERS = 32
HIDDEN_DIM = 4096

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    token=hf_token,
    attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  Model loaded: {MODEL_NAME}")

# ============================================================
# 5. Run layer sweep for each concept
# ============================================================
from safety_compass import ConceptDirectionExtractor

model_config = {
    "model_name": MODEL_NAME,
    "num_layers": NUM_LAYERS,
    "hidden_dim": HIDDEN_DIM,
}

CONCEPTS = [
    {
        "name": "refusal",
        "pairing_strategy": "arditi",
        "contrastive_pairs_file": "refusal.jsonl",
    },
    {
        "name": "sycophancy",
        "pairing_strategy": "caa",
        "contrastive_pairs_file": "sycophancy.jsonl",
    },
    {
        "name": "deception",
        "pairing_strategy": "caa",
        "contrastive_pairs_file": "deception.jsonl",
    },
]

results = {}

for concept in CONCEPTS:
    name = concept["name"]
    print(f"\n{'=' * 60}")
    print(f"Layer sweep: {name}")
    print(f"{'=' * 60}")

    extractor = ConceptDirectionExtractor(
        model=model,
        tokenizer=tokenizer,
        concept_config=concept,
        model_config=model_config,
    )
    extractor.load_pairs(base_dir=PAIRS_DIR)

    sweep_start = time.time()
    result = extractor.find_best_layer()
    sweep_time = time.time() - sweep_start

    results[name] = {
        "best_layer": int(result["best_layer"]),
        "best_auroc": float(result["best_auroc"]),
        "sweep_seconds": round(sweep_time, 1),
    }

    print(f"  Best layer: {result['best_layer']}")
    print(f"  Best AUROC: {result['best_auroc']:.4f}")
    print(f"  Sweep time: {sweep_time:.1f}s")

    if "all_aurocs" in result:
        aurocs_path = os.path.join(WORK_DIR, f"{name}_layer_aurocs.json")
        with open(aurocs_path, "w") as f:
            aurocs_data = {
                str(k): float(v) for k, v in result["all_aurocs"].items()
            } if isinstance(result["all_aurocs"], dict) else [
                float(v) for v in result["all_aurocs"]
            ]
            json.dump(aurocs_data, f, indent=2)

# ============================================================
# 6. Save results
# ============================================================
output_path = os.path.join(WORK_DIR, "best_layers.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

total_time = time.time() - start_time

print("\n" + "=" * 60)
print("LAYER SWEEP RESULTS")
print("=" * 60)
print(f"  Model: {MODEL_NAME}")
for name, r in results.items():
    separable = "YES" if r["best_auroc"] >= 0.80 else "WEAK"
    print(f"  {name}: layer {r['best_layer']}, AUROC {r['best_auroc']:.4f} ({separable})")
print(f"  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"  Results saved to: {output_path}")
print("=" * 60)
