"""Build LLM fine-tuning datasets (JSONL) from a manually annotated patient table.

Reads one "patients_tableN.xlsm" workbook from data/1_manual_table/ (one sheet
per patient, last sheet assumed to be a non-patient summary/legend sheet and
is dropped), and for every note row produces five parallel JSONL datasets
under data/2_input_model/, one per prompting strategy:

    with_cot     - system prompt asks for a chain-of-thought "thought" field
                   plus the extracted "extraction" JSON (sparse: only
                   positive findings).
    without_cot  - same sparse extraction, but no chain-of-thought.
    dry_run      - baseline: minimal instructions + explicit list of every
                   possible feature name, sparse extraction.
    no_prompt    - no system prompt at all, sparse extraction.
    unknow       - every feature from the table is included, with "unknown"
                   for anything not mentioned in the note (dense extraction).

Each output line is a JSON object: {"metadata": {...}, "messages": [...]}.

How to run:
    Edit `table_number` below to match the target patients_table<N>.xlsm
    file, then run this script directly:

        python generate_json.py

Input:
    data/1_manual_table/patients_table<table_number>.xlsm
Outputs:
    data/2_input_model/with_cot/training_data_with_cot<table_number>.jsonl
    data/2_input_model/without_cot/training_data_without_cot<table_number>.jsonl
    data/2_input_model/dry_run/training_data_dry_run<table_number>.jsonl
    data/2_input_model/no_prompt/training_data_no_prompt<table_number>.jsonl
    data/2_input_model/unknow/training_data_unknow<table_number>.jsonl
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
import pandas as pd
import json
from pathlib import Path
from dateutil.relativedelta import relativedelta


#TO RUN:
table_number = "6"


# -----------------------------------------------------------------------------
# 1. Path Configuration
# -----------------------------------------------------------------------------
script_folder = Path(__file__).resolve().parent  # .resolve() turns the relative script path into an absolute one; .parent keeps the folder containing this file
project_root = script_folder.parent  # move up one level to get the repo root

table_path =  project_root/"data"/"1_manual_table"/f"patients_table{table_number}.xlsm"
json_path_with_cot = project_root/"data"/"2_input_model"/"with_cot"/f"training_data_with_cot{table_number}.jsonl"
json_path_without_cot = project_root/"data"/"2_input_model"/"without_cot"/f"training_data_without_cot{table_number}.jsonl"
json_path_dry_run = project_root/"data"/"2_input_model"/"dry_run"/f"training_data_dry_run{table_number}.jsonl"
json_path_no_prompt = project_root/"data"/"2_input_model"/"no_prompt"/f"training_data_no_prompt{table_number}.jsonl"
json_path_unknow = project_root/"data"/"2_input_model"/"unknow"/f"training_data_unknow{table_number}.jsonl"

# Create the output folders, which do not exist in a fresh clone
for json_path in (json_path_with_cot, json_path_without_cot, json_path_dry_run, json_path_no_prompt, json_path_unknow):
    json_path.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Data Preprocessing
# -----------------------------------------------------------------------------
sheets = pd.read_excel(table_path, sheet_name=None)  # sheet_name=None loads every sheet into a dict {sheet_name: DataFrame}
patients = list(sheets.keys())[0:-1]  # all sheets except the last one, which is assumed to be a non-patient summary/legend sheet
print("\n")
print(f"Patients found : {len(patients)} ")
print("\n")

# Merge patients sheets together - adding sheet name as source for backup patient ID
patients_sheets = []
for patient in patients:
    df = sheets[patient].copy()
    df['_source_sheet'] = patient  # Add sheet name as source backup
    patients_sheets.append(df)
sheet = pd.concat(patients_sheets, ignore_index=True)  # ignore_index=True renumbers rows continuously (0..N) instead of restarting per sheet

# Extract and keep Patient ID, Note Date, Note Month and source sheet columns before dropping.
# NOTE: columns[1..3] are positional and depend on the fixed layout of the
# patients_table*.xlsm sheets (Patient ID, Note Date, Note Month).
metadata_df = sheet[[sheet.columns[1], sheet.columns[2], sheet.columns[3], '_source_sheet']].copy()

sheet = sheet.drop(sheet.columns[:4], axis=1)  # drop the 4 leading metadata columns (kept separately in metadata_df); axis=1 = columns, not rows

# Remove source sheet column if it exists
if "_source_sheet" in sheet.columns:
    sheet = sheet.drop("_source_sheet", axis=1)

# Remove "Annotated" column if it exists
if "Annotated" in sheet.columns:
    sheet = sheet.drop("Annotated", axis=1)

sheet = sheet.astype(object).where(pd.notnull(sheet), None)  # normalize blanks to None; cast to object dtype to avoid mixed number/text coercion issues
sheet = sheet.dropna(subset=["Raw Note Text"])  # drop rows with no clinical note text

# Keep only metadata for rows that remain and reset index
metadata_df = metadata_df.loc[sheet.index].reset_index(drop=True)
sheet = sheet.reset_index(drop=True)
print(sheet)
print("\n")

# First, rebuild full metadata with patient ID for month calculation
# Extract Patient ID for each row
patient_ids_list = []
for idx in range(len(metadata_df)):
    patient_id_val = metadata_df.iloc[idx, 0]
    source_sheet = metadata_df.iloc[idx, 3]

    if pd.notna(patient_id_val):
        try:
            patient_id = str(int(patient_id_val))
        except (ValueError, TypeError):
            patient_id = str(patient_id_val)
    else:
        patient_id = str(source_sheet) if pd.notna(source_sheet) else "Unknown"

    patient_ids_list.append(patient_id)

# Add patient IDs to metadata for grouping
metadata_df_with_ids = metadata_df.copy()
metadata_df_with_ids['_patient_id'] = patient_ids_list

# Calculate correct note_month based on first visit date for each patient
def calculate_months_from_first_visit(dates_series):
    """Return, for each date in `dates_series`, the number of whole months
    elapsed since the earliest date in the series (i.e. months since the
    patient's first recorded visit). Unparseable/missing dates map to 0.
    """
    dates = pd.to_datetime(dates_series, errors='coerce')
    first_date = dates.min()

    if pd.isna(first_date):
        return [0] * len(dates_series)

    # Calculate months for each date
    months_list = []
    for curr_date in dates:
        if pd.isna(curr_date):
            months_list.append(0)
        else:
            delta = relativedelta(curr_date, first_date)
            months_list.append(delta.years * 12 + delta.months)

    return months_list

# Group by patient and calculate corrected months
corrected_months_dict = {}

for patient_id in metadata_df_with_ids['_patient_id'].unique():
    mask = metadata_df_with_ids['_patient_id'] == patient_id
    group_indices = metadata_df_with_ids[mask].index.tolist()

    # Get dates for this patient using iloc (column 1 is note_date)
    group_dates = metadata_df.iloc[group_indices, 1]

    months_for_group = calculate_months_from_first_visit(group_dates)

    for idx, month in zip(group_indices, months_for_group):
        corrected_months_dict[idx] = month

metadata_df['_corrected_month'] = metadata_df.index.map(corrected_months_dict)

print("\n✓ Note months recalculated based on first patient visit")
print("\n")


# -----------------------------------------------------------------------------
# 3. Json Construction
# -----------------------------------------------------------------------------
# Prompt Strategy : Sparse + Synthetic CoT
system_prompt_with_cot = (
    "You are an expert Orthodontist Assistant. "
    "First, analyze the clinical note in the 'thought' field, citing evidence from the text. "
    "Then, extract the positive findings into the 'extraction' field as JSON."
)

system_prompt_without_cot = (
    "You are an expert Orthodontist Assistant. "
    "Extract the positive findings from the clinical note as JSON."
)

system_prompt_dry_run = (
    "You are an expert Orthodontist Assistant. Extract orthodontic findings from the clinical note as JSON.\n\n"
    "Format your response as: {\"extraction\": {...}}\n\n"
    "Potential features to extract:\n"
    "Patient_ID, Note_Date, Note_Month, Annotated, Upper Wire Size, Upper Wire Material, Lower Wire Size, "
    "Lower Wire Material, Changed Upper Arch Wire, Changed Lower Arch Wire, Ligature Method, Oral Hygiene, "
    "Elastic Pattern Left, Right Canine Class, Left Canine Class, Right Molar Class, Left Molar Class, "
    "Class II elastic, Elastic Pattern Right, Compliance, Overjet (mm), Elastic Type Left, Elastic Type Right, "
    "Overbite (mm), Debonded Bracket, Lower Retainer, Emergency Type, Upper Retainer, Space closure sliding mechanics, "
    "Photos taken, Upper Arch Bends, Class I elastic, Class III elastic, Appliance, Lower Arch Bends, Retainer Check, "
    "Xrays taken, Intra Oral Scanning Taken, EMERGENCY, Lower Arch Reverse Curve of Spee, Bracket OR BAND Repositioning, "
    "Open Spring, Upper Bonding, IPR, Re-tie Appointment, Lower Bonding, Posterior Bite Turbos, Cross Elastic, "
    "Upper Arch Accentuated Curve of Spee, Upper Debond, Lower Debond, TADs, Prescription and Bracket Slot, TMJ symptoms, "
    "Enameloplasty, Referral, Unilateral Posterior Crossbite, Extractions, TPA, Space closure loop mechanics, Upper Banding, "
    "Relapse, Upper Active movement, Lower Active movement, Closed Spring, Lower Banding, Patient ID.1, NiTi Closing Spring, "
    "Anterior Bite Turbos, TADs.1, Upper Arch Reverse Curve of Spee, Maxillary Expander, LLHA, Anterior Crossbite, "
    "Debonded Bracket/Band, Lower Arch Curve of Spee, Intrusion Arch, Bilateral Posterior Crossbite, "
    "Lower Arch Accentuated Curve of Spee, Active Traction, Active Tooth Traction, Mandibular Advancement Appliance, "
    "Teeth Pain, Arch Coordination\n\n"
    "Rules:\n"
    "- Only include features that are explicitly mentioned or clearly implied in the note\n"
    "- Use boolean (true) for binary features when found, or the actual value\n"
    "- Omit features not mentioned in the note"
)

system_prompt_no_prompt = ""

system_prompt_unknow = (
    "You are an expert Orthodontist Assistant. "
    "Extract orthodontic findings from the clinical note as JSON. "
    "For each feature mentioned in the table columns, include its value if found, or 'unknown' if not mentioned."
)

training_data_with_cot = []
training_data_without_cot = []
training_data_dry_run = []
training_data_no_prompt = []
training_data_unknow = []

# Loop patients
for idx, row in sheet.iterrows():
    note = row["Raw Note Text"]  # Extract patient note

    # Extract Patient ID, Note Date and Note Month from metadata
    patient_id_val = metadata_df.iloc[idx, 0]
    source_sheet = metadata_df.iloc[idx, 3]  # Get source sheet as backup

    if pd.notna(patient_id_val):
        try:
            patient_id = str(int(patient_id_val))
        except (ValueError, TypeError):
            patient_id = str(patient_id_val)
    else:
        # If no patient_id, use source sheet name as backup
        patient_id = str(source_sheet) if pd.notna(source_sheet) else "Unknown"

    # Keep full date, remove only the time part (00:00:00)
    note_date_str = str(metadata_df.iloc[idx, 1])
    note_date = note_date_str.replace(' 00:00:00', '') if ' 00:00:00' in note_date_str else note_date_str

    # Extract Note Month
    note_month_val = metadata_df.iloc[idx, 2]
    corrected_month = metadata_df.iloc[idx, 4]  # Get corrected month from new column

    if pd.notna(corrected_month):
        note_month = str(int(corrected_month))
    elif pd.notna(note_month_val):
        # Handle both numeric and Timestamp types
        try:
            note_month = str(int(note_month_val))
        except (ValueError, TypeError):
            # If it's a Timestamp, extract the month
            if hasattr(note_month_val, 'month'):
                note_month = str(note_month_val.month)
            else:
                note_month = str(note_month_val)
    else:
        note_month = "0"

    # Create enhanced user message (without metadata in content)
    enhanced_note = f"{note}"

    raw_features = row.drop("Raw Note Text").to_dict()  # Extract patient features
    sparse_features = {}
    unknow_features = {}
    thought_list = []

    # Loop Value (WITHOUT adding metadata as features)
    for key, value in raw_features.items():
        if pd.notna(value):  # Sparse: keep only positive (non-missing) values
            # Convert Timestamp to string for JSON serialization
            if hasattr(value, 'isoformat'):  # Check if it's a Timestamp or datetime
                value = value.isoformat()

            if value != 1.0:  # pandas reads Excel TRUE/checkbox cells as the float 1.0
                sparse_features[key] = value
            else:
                sparse_features[key] = True
            thought_list.append(f"Found evidence for {key}.")  # one thought sentence per positive value
            unknow_features[key] = value
        else:
            # For unknow format: add "unknown" for missing values
            unknow_features[key] = "unknown"

    if not thought_list:  # Rare case but security (Patient has no value)
        thoughts = "No specific orthodontic procedures detected."
    else:  # Common case
        thoughts = "Analysis: " + " ".join(thought_list)

    # Output = thought + value
    final_output_with_cot = {"thought": thoughts, "extraction": sparse_features}
    final_output_without_cot = {"extraction": sparse_features}
    final_output_no_prompt = {"extraction": sparse_features}
    final_output_unknow = {"extraction": unknow_features}

    # Convert Python Object into text (Json String)
    json_response_with_cot = json.dumps(final_output_with_cot, ensure_ascii=False)
    json_response_without_cot = json.dumps(final_output_without_cot, ensure_ascii=False)
    json_response_no_prompt = json.dumps(final_output_no_prompt, ensure_ascii=False)
    json_response_unknow = json.dumps(final_output_unknow, ensure_ascii=False)

    # Chat format WITH CoT (used by OpenAi etc ...)
    training_data_with_cot.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "system", "content": system_prompt_with_cot},
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_with_cot},
        ]
    })

    # Chat format WITHOUT CoT
    training_data_without_cot.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "system", "content": system_prompt_without_cot},
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_without_cot},
        ]
    })

    # Chat format for DRY RUN (baseline with minimal prompt)
    training_data_dry_run.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "system", "content": system_prompt_dry_run},
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_without_cot},
        ]
    })

    # Chat format NO PROMPT (no system prompt)
    training_data_no_prompt.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_no_prompt},
        ]
    })

    # Chat format UNKNOW (with unknown for missing features)
    training_data_unknow.append({
        "metadata": {
            "patient_id": patient_id,
            "note_date": note_date,
            "note_month": note_month
        },
        "messages": [
            {"role": "system", "content": system_prompt_unknow},
            {"role": "user", "content": enhanced_note},
            {"role": "assistant", "content": json_response_unknow},
        ]
    })

# -----------------------------------------------------------------------------
# 4. Json save
# -----------------------------------------------------------------------------
print(f"TABLE NUMBER : {table_number}\n")
with open(json_path_with_cot, "w", encoding="utf-8") as json_file:
    for item in training_data_with_cot:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved WITH CoT as {json_path_with_cot}")

with open(json_path_without_cot, "w", encoding="utf-8") as json_file:
    for item in training_data_without_cot:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved WITHOUT CoT as {json_path_without_cot}")

with open(json_path_dry_run, "w", encoding="utf-8") as json_file:
    for item in training_data_dry_run:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved DRY RUN (Baseline) as {json_path_dry_run}")

with open(json_path_no_prompt, "w", encoding="utf-8") as json_file:
    for item in training_data_no_prompt:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved NO PROMPT as {json_path_no_prompt}")

with open(json_path_unknow, "w", encoding="utf-8") as json_file:
    for item in training_data_unknow:
        json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"Saved UNKNOW as {json_path_unknow}")
