"""Build a side-by-side manual-vs-model comparison table from model output.

Reads a JSONL file of model predictions (produced by the inference step, one
record per note, each holding the original note, the ground-truth
"extraction" JSON, and the model's predicted "extraction" JSON) and flattens
it into a long-format Excel table with one row per (patient, note, feature)
pair, for easy manual review / metric computation in Excel.

How to run:
    Edit `table_number` and `mode` below to select the input file, then run:

        python generate_table.py

Input:
    data/3_output_model/<mode>/extraction_llama_<mode><table_number>_all.jsonl
    Each line is expected to be a JSON object with:
      - "original_note": the raw clinical note text
      - "original": a JSON string like {"extraction": {...}} (ground truth)
      - "prediction": a JSON string like {"extraction": {...}} (model output)
      - "metadata": optional {"patient_id", "note_date", "note_month"}
Output:
    data/4_model_table/<mode>/extraction_features_<mode><table_number>_all.xlsx
    Columns: Patient_ID, Note_Date, Note_Month, Feature, Manual, Model
"""
import json
import pandas as pd
from pathlib import Path


#TO RUN:
table_number = 6
mode = "no_prompt"  # Options: "with_cot", "without_cot", "dry_run", "no_prompt"


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent  # .resolve() turns the relative script path into an absolute one; .parent keeps the folder containing this file
project_root = script_folder.parent  # move up one level to get the repo root

# Adapt paths based on mode
json_path = project_root/"data"/"3_output_model"/f"{mode}"/f"extraction_llama_{mode}{table_number}_all.jsonl"
table_path = project_root/"data"/"4_model_table"/f"{mode}"/f"extraction_features_{mode}{table_number}_all.xlsx"

# Ensure output directory exists
table_path.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Data preprocessing
# -----------------------------------------------------------------------------
data_row = []

with open(json_path, "r", encoding="utf-8") as json_file:
    for i, line in enumerate(json_file):
        if not line.strip(): continue
        try:
            record = json.loads(line)

            # Notes
            note = record.get('original_note')

            # truth
            truth_str = record.get('original')
            truth_json = json.loads(truth_str)
            truth = truth_json.get("extraction")

            # prediction
            prediction_str = record.get('prediction')
            prediction_json = json.loads(prediction_str)
            prediction = prediction_json.get("extraction")

            # Extract Patient_ID, Note_Date, Note_Month from metadata if available, else from truth
            metadata = record.get("metadata", {})
            patient_id = metadata.get("patient_id") or truth.get("Patient_ID", "Unknown")
            note_date = metadata.get("note_date") or truth.get("Note_Date", "Unknown")
            note_month = metadata.get("note_month") or truth.get("Note_Month", "Unknown")

            # Combine keys from both truth and prediction so features present
            # in only one side still get a row (with the other side blank).
            all_features = set(truth.keys()).union(set(prediction.keys()))

            # Patient_ID/Note_Date/Note_Month are handled separately above;
            # drop them here so they don't also appear as "Feature" rows.
            for meta_key in ["Patient_ID", "Note_Date", "Note_Month"]:
                all_features.discard(meta_key)

            for feature in all_features:
                val_manual = truth.get(feature)
                val_model = prediction.get(feature)

                def is_empty(v):
                    """A value counts as "not reported" for this feature."""
                    return v in (None, "", [], {})

                # Skip the feature entirely if both manual and model agree it's absent
                # (keeps the table to only meaningful/disagreement rows).
                if is_empty(val_manual) and is_empty(val_model):
                    continue

                data_row.append({
                    "Patient_ID": patient_id,
                    "Note_Date": note_date,
                    "Note_Month": note_month,
                    "Feature": feature,
                    "Manual": val_manual,
                    "Model": val_model
                })

        except Exception as e:
            print(f"Error line: {e}")

# -----------------------------------------------------------------------------
# 3. Save
# -----------------------------------------------------------------------------
df = pd.DataFrame(data_row)
# Col Organisation: Patient_ID, Note_Date, Note_Month, Feature, Manual, Model
columns_order = ["Patient_ID", "Note_Date", "Note_Month", "Feature", "Manual", "Model"]

df = df[columns_order]
df.to_excel(table_path, index=False)
print(f"Save file {table_path}")
