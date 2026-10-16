"""Report how often each feature appears across a JSONL training dataset.

Reads a generate_json.py JSONL file and counts, for every feature key found
in any assistant "extraction" JSON, how many records contain it. Prints a
table sorted by frequency (most common first), plus the average
representation percentage across all features - useful for spotting rare
or over/under-represented features before training.

How to run:
    Edit `table_number` and `mode` below, then run:

        python print_representation.py

Input:
    data/2_input_model/<mode>/training_data_<mode><table_number>.jsonl
Output:
    Console output only (no files written).
"""
import json
from pathlib import Path
from collections import Counter


#TO RUN:
table_number = "6"
mode = "no_prompt"  # Change to "with_cot" or "without_cot" to compare

print("-" * 95)
print(f" {mode}, Table {table_number} - REPRESENTATION ANALYSIS")
print("-" * 95)


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent
json_path = str(project_root / "data"/"2_input_model"/f"{mode}"/f"training_data_{mode}{table_number}.jsonl")

# -----------------------------------------------------------------------------
# 2. Process Loop
# -----------------------------------------------------------------------------
feature_counts = Counter()
total_records = 0
valid = 0
corrupted = 0

with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():  # Ignore empty line
            continue
        try:
            record = json.loads(line)
            total_records += 1

            # Feature counting
            for msg in record.get("messages", []):
                if msg.get("role") == "assistant":
                    content = json.loads(msg.get("content", "{}"))
                    extraction = content.get("extraction", {})
                    for feature in extraction.keys():
                        feature_counts[feature] += 1
            valid += 1
        except:
            corrupted += 1

print(f"\n✓ Valid: {valid}  |  ✗ Corrupted: {corrupted}")

print("\n" + "="*70)
print("FEATURES ANALYSIS (Representation)")
print("="*70)

print("\nFEATURE" + " "*50 + "| COUNT | COUNT %")
print("-" * 70)

for feature, count in feature_counts.most_common():
    percentage = (count * 100) // total_records if total_records > 0 else 0  # integer percentage, truncated (not rounded)
    print(f"{feature:<55} | {count:5} | {percentage:3} %")

print("-" * 70)
print(f"TOTAL RECORDS: {total_records}")
print(f"\n✓ UNIQUE FEATURES DETECTED: {len(feature_counts)}")

# Calculate average representation percentage
if feature_counts:
    average_percentage = sum((count * 100) // total_records for count in feature_counts.values()) / len(feature_counts)
    print(f"\n📊 AVERAGE REPRESENTATION %: {average_percentage:.2f}%")

print("\n" + "-" * 95)
print(f" {mode}, Table {table_number} - REPRESENTATION ANALYSIS")
print("-" * 95)
