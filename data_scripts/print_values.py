"""Report the distribution of values taken by each feature in a JSONL dataset.

Reads a generate_json.py JSONL file and, for every feature key found in any
assistant "extraction" JSON, counts how many times each distinct value
occurs (values are lowercased/stripped, and lists are sorted, so equivalent
values are grouped together). Prints a full breakdown per feature and also
writes the same report to a .txt file next to this script, for offline
review of label/value consistency (e.g. spotting inconsistent free-text
values that should be normalized).

How to run:
    Edit `table_number` and `mode` below, then run:

        python print_values.py

Input:
    data/2_input_model/<mode>/training_data_<mode><table_number>.jsonl
Output:
    Console output, and a copy of the same report written to
    data_scripts/feature_values_analysis_<mode>_table<table_number>.txt
"""
import json
from pathlib import Path
from collections import defaultdict

# TO RUN:
table_number = "6"
mode = "no_prompt"  # Change to "with_cot" or "without_cot"

# Path Configuration
script_folder = Path(__file__).resolve().parent
project_root = script_folder.parent
output_log = project_root / "data_scripts" / f"feature_values_analysis_{mode}_table{table_number}.txt"

# Open file for output
output_file = open(output_log, "w", encoding="utf-8")

def print_to_file(message=""):
    """Print to both console and file"""
    print(message)
    output_file.write(message + "\n")
    output_file.flush()

print_to_file("-" * 95)
print_to_file(f"FEATURE VALUES ANALYSIS - {mode}, Table {table_number}")
print_to_file("-" * 95)

json_path = str(project_root / "data" / "2_input_model" / f"{mode}" / f"training_data_{mode}{table_number}.jsonl")

# Dictionary to store values count for each feature
feature_values = defaultdict(lambda: defaultdict(int))
total_records = 0
valid_records = 0
corrupted_records = 0

print_to_file(f"\nAnalyzing: {json_path}\n")

with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            total_records += 1

            # Extract features from messages
            for msg in record.get("messages", []):
                if msg.get("role") == "assistant":
                    try:
                        content = json.loads(msg.get("content", "{}"))
                        extraction = content.get("extraction", {})

                        # For each feature, count its values
                        for feature, value in extraction.items():
                            # Normalize the value to a comparable string so that
                            # e.g. "True"/"true" or differently-ordered lists count as the same value.
                            if isinstance(value, list):
                                value_str = str(sorted([str(v).lower().strip() for v in value]))
                            elif value is None:
                                value_str = "None"
                            else:
                                value_str = str(value).lower().strip()

                            feature_values[feature][value_str] += 1

                        valid_records += 1
                    except:
                        corrupted_records += 1
        except:
            corrupted_records += 1

print_to_file(f"\nTotal Records Processed: {total_records}")
print_to_file(f"Valid Records: {valid_records}")
print_to_file(f"Corrupted Records: {corrupted_records}")

# Display results
print_to_file("\n" + "=" * 95)
print_to_file("FEATURE VALUES ANALYSIS")
print_to_file("=" * 95)

# Build a flat list of per-(feature, value) rows. NOTE: `results` is currently
# collected but never written out (no CSV export is implemented) - the
# console/.txt report above is the only output actually produced. Kept as-is
# in case CSV export is added later; safe to remove if not needed.
results = []
for feature in sorted(feature_values.keys()):
    values_dict = feature_values[feature]
    total_occurrences = sum(values_dict.values())
    unique_values = len(values_dict)

    print_to_file(f"\nFeature: {feature}")
    print_to_file(f"   Total occurrences: {total_occurrences}")
    print_to_file(f"   Unique values: {unique_values}")
    print_to_file(f"   {'-' * 90}")

    # Sort by frequency (descending)
    sorted_values = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)

    for value, count in sorted_values:
        percentage = (count / total_occurrences) * 100
        print_to_file(f"   {value[:70]:70} | {count:5d} ({percentage:5.1f}%)")

        # Store for CSV export
        results.append({
            "Feature": feature,
            "Value": value,
            "Count": count,
            "Percentage": f"{percentage:.1f}%",
            "Unique_Values": unique_values,
            "Total_Occurrences": total_occurrences
        })

print_to_file("\n" + "=" * 95)
print_to_file("ANALYSIS COMPLETE")
print_to_file("=" * 95)

output_file.close()
print(f"\nAll results saved to: {output_log}")
