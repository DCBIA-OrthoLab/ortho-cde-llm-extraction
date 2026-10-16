"""Print quick headline counts for a generate_json.py JSONL dataset.

Reads one training JSONL file and prints the number of unique patients,
the number of records (notes), and the total number of extracted features
across all records - a fast sanity check without opening the file.

How to run:
    Edit `table_number` and `mode` below, then run:

        python print_numbers.py

Input:
    data/2_input_model/<mode>/training_data_<mode><table_number>.jsonl
Output:
    Console output only (no files written).
"""
import json
from pathlib import Path

# TO RUN:
table_number = "6"
mode = "no_prompt"  # "with_cot", "without_cot", or "dry_run"

# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent

json_path = project_root / "data" / "2_input_model" / mode / f"training_data_{mode}{table_number}.jsonl"

# Check if file exists
if not json_path.exists():
    print(f"❌ File not found: {json_path}")
    exit(1)

# -----------------------------------------------------------------------------
# 2. Counter initialization
# -----------------------------------------------------------------------------
unique_patients = set()  # a set keeps only unique patient IDs
total_records = 0
total_features = 0

# -----------------------------------------------------------------------------
# 3. Read and process the file
# -----------------------------------------------------------------------------
print(f"Reading data from: {json_path} ...\n")

with open(json_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            # Parse the JSON line
            record = json.loads(line)

            # 1. Count the record (1 line = 1 record)
            total_records += 1

            # 2. Count the patient (added to the set)
            patient_id = record.get("metadata", {}).get("patient_id")
            if patient_id:
                unique_patients.add(patient_id)

            # 3. Count the annotated features
            # Look for the assistant message, which holds the extraction
            for message in record.get("messages", []):
                if message.get("role") == "assistant":
                    content_str = message.get("content", "{}")
                    try:
                        # 'content' is a JSON string, needs to be parsed
                        content_json = json.loads(content_str)
                        extraction = content_json.get("extraction", {})

                        # Add the number of extracted features (dict size)
                        total_features += len(extraction)

                    except json.JSONDecodeError:
                        print(f"⚠️ Could not decode 'content' for patient {patient_id}")

        except json.JSONDecodeError:
            print("⚠️ Invalid JSON line skipped.")

# -----------------------------------------------------------------------------
# 4. Display results
# -----------------------------------------------------------------------------
print("=" * 40)
print("📊 DATA SUMMARY")
print("=" * 40)
print(f"Unique patients   : {len(unique_patients)}")
print(f"Records (notes)   : {total_records}")
print(f"Annotated features: {total_features}")
print("=" * 40)
