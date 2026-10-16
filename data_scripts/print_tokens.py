"""Report tokenized length statistics for a JSONL training dataset.

Reads a generate_json.py JSONL file, renders each record's chat messages
through the Phi-3.5 chat template, tokenizes the result, and prints the
average/min/max token count per record - used to check that examples fit
within the target model's context window before training.

How to run:
    Edit `table_number` and `mode` below, then run:

        python print_tokens.py

Note: downloads/loads the "unsloth/Phi-3.5-mini-instruct" tokenizer from
Hugging Face on first run (requires the `transformers` package and network
access, or a local HF cache).

Input:
    data/2_input_model/<mode>/training_data_<mode><table_number>.jsonl
Output:
    Console output only (no files written).
"""
import json
from pathlib import Path
from transformers import AutoTokenizer


#TO RUN:
table_number = "6"
mode = "no_prompt"  # Change to "with_cot" or "without_cot" to compare

print("-" * 95)
print(f" {mode}, Table {table_number} - TOKEN ANALYSIS")
print("-" * 95)


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent
json_path = str(project_root / "data"/"2_input_model"/f"{mode}"/f"training_data_{mode}{table_number}.jsonl")

# Load tokenizer for Phi-3.5
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/Phi-3.5-mini-instruct")  # NOTE: hardcoded to Phi-3.5; update if targeting a different base model

# -----------------------------------------------------------------------------
# 2. Process Loop
# -----------------------------------------------------------------------------
total_records = 0
valid = 0
corrupted = 0
token_stats = {
    "record_tokens": [],
    "min_tokens": float('inf'),
    "max_tokens": 0
}

with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():  # Ignore empty line
            continue
        try:
            record = json.loads(line)
            total_records += 1

            # Extract all messages and format them
            conversation = record.get("messages", [])
            formatted_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False
            )

            # Count tokens for this record
            tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
            num_tokens = len(tokens)
            token_stats["record_tokens"].append(num_tokens)
            token_stats["min_tokens"] = min(token_stats["min_tokens"], num_tokens)
            token_stats["max_tokens"] = max(token_stats["max_tokens"], num_tokens)

            valid += 1
        except:
            corrupted += 1

print(f"\n✓ Valid: {valid}  |  ✗ Corrupted: {corrupted}")

# ============ TOKENS ANALYSIS ============
print("\n" + "="*70)
print("TOKENS ANALYSIS (Context Window)")
print("="*70)
if token_stats["record_tokens"]:
    avg_tokens = sum(token_stats["record_tokens"]) / len(token_stats["record_tokens"])
    print(f"Average Tokens/Record:  {avg_tokens:.1f}")
    print(f"Min Tokens:             {token_stats['min_tokens']}")
    print(f"Max Tokens:             {token_stats['max_tokens']}")

print("\n" + "-" * 95)
print(f" {mode}, Table {table_number} - TOKEN ANALYSIS")
print("-" * 95)
