"""
Score a model's extraction predictions (written by run_model.py) against
the ground-truth extraction JSON, both exactly (precision/recall/F1 per
feature) and semantically (cosine similarity of sentence embeddings for
non-exact-match values), then export the results to an Excel workbook.

Invocation:
    python eval_model.py
        Runs main() with the hardcoded defaults in the __main__ block below
        (edit that call, or import `main` from another script, e.g.
        advanced/run_pipeline.py, to change table_number/mode/model_type).

Reads:
    data/3_output_model/{mode}/extraction_{...}{mode}{table_number}.jsonl
        Produced by run_model.py. Each line must have "original" (ground
        truth assistant message, a JSON string containing an "extraction"
        object) and "prediction" (model output - either a dict already, or
        a JSON string, again containing an "extraction" object except for
        mode == "tmj" whose "extraction" key nests one level deeper).

Writes:
    data/5_results/{mode}/result[_llama|_qwen]_{mode}{table_number}.xlsx
        Excel workbook with three sheets: "Features" (per-feature precision/
        recall/F1/semantic-score), "Global Metrics" (simple + weighted
        averages), "Summary" (run metadata). Numbers use French decimal
        formatting (comma instead of dot) since the intended audience
        reviews these files in French Excel.
"""

import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm


def main(table_number: int, mode: str, model_type: str = "phi"):
    """
    Main function for evaluating model predictions.
    
    Args:
        table_number: Table number (1-5)
        mode: Training mode (no_prompt, with_cot, without_cot, tmj, dry_run)
        model_type: Model to evaluate ("phi" or "llama")
    """
    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)

    # --- Configuration ---
    project_root = Path(__file__).resolve().parents[1]
    
    # Determine input file based on model type. The names match the ones
    # run_model.py writes for each model type.
    if model_type == "llama":
        json_path = project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_llama_{mode}{table_number}.jsonl"
    elif model_type == "qwen":
        json_path = project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_qwen_{mode}{table_number}.jsonl"
    else:
        json_path = project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_{mode}{table_number}_verif.jsonl"

    results_dir = project_root / "data" / "5_results" / f"{mode}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Input file: {json_path}")
    print(f"📁 Output directory: {results_dir}")
    print(f"\n⏳ Loading semantic model...")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✓ Semantic model loaded")

    def get_pairs(d):
        """
        Turns an extraction dict into a set of normalized (key, value)
        pairs so two dicts can be compared with set operations for
        TP/FP/FN below. List values are sorted before stringifying so pair
        equality doesn't depend on list order.
        """
        if not d: return set()
        pairs = set()
        for k, v in d.items():
            k_norm = str(k).lower().strip()
            if isinstance(v, list):
                v_norm = str(sorted([str(x).lower().strip() for x in v]))
            else:
                v_norm = str(v).lower().strip()

            # Skip "unknown" values - they are not relevant to the evaluation
            if v_norm != "unknown":
                pairs.add((k_norm, v_norm))
        return pairs

    # --- 1. Collect the data ---
    print(f"\n📖 Reading input file...")
    results = []
    total_lines = 0
    record_data = []  # Store original/prediction dicts for semantic scoring
    skipped = 0

    with open(json_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    for i, line in tqdm(enumerate(lines), total=len(lines), desc="Processing records", unit=" records"):
        total_lines += 1
        try:
            record = json.loads(line)
        except Exception as e:
            print(f"⚠️  Parsing error on line {i+1}: {e}")
            skipped += 1
            continue
        
        try:
            orig_dict = json.loads(record["original"]).get("extraction", {})
            # Handle prediction as either dict or JSON string
            if isinstance(record["prediction"], dict):
                # For tmj, the structure is nested with "extraction" inside
                if mode == "tmj":
                    pred_dict = record["prediction"].get("extraction", {})
                else:
                    pred_dict = record["prediction"]
            else:
                pred_dict = json.loads(record["prediction"]).get("extraction", {})
        except Exception as e:
            print(f"⚠️  Extraction error on line {i+1}: {e}")
            skipped += 1
            continue

        record_data.append((orig_dict, pred_dict))
        s_orig = get_pairs(orig_dict)
        s_pred = get_pairs(pred_dict)

        # TP: intersection of the pairs (same key + value)
        for k, v in s_orig & s_pred:
            results.append({'feature': k, 'type': 'tp', 'val_orig': v, 'val_pred': v, 'key': k})

        # FP: in the prediction but not in the original
        for k, v in s_pred - s_orig:
            results.append({'feature': k, 'type': 'fp', 'val_orig': orig_dict.get(k, ""), 'val_pred': v, 'key': k})

        # FN: in the original but not in the prediction
        for k, v in s_orig - s_pred:
            results.append({'feature': k, 'type': 'fn', 'val_orig': v, 'val_pred': pred_dict.get(k, ""), 'key': k})
        
        # FN - empty values in the prediction where the original has a value
        for k in orig_dict.keys():
            k_norm = str(k).lower().strip()
            v_orig_norm = str(orig_dict[k]).lower().strip() if orig_dict[k] else ""
            
            # Check whether the key exists in pred_dict but is empty
            if k in pred_dict:
                v_pred = pred_dict[k]
                v_pred_norm = str(v_pred).lower().strip() if v_pred else ""
                
                # Original has a value but the prediction is empty -> FN
                if v_orig_norm and not v_pred_norm:
                    results.append({'feature': k_norm, 'type': 'fn', 'val_orig': v_orig_norm, 'val_pred': "", 'key': k_norm})

    df = pd.DataFrame(results)

    # Sanity check
    if df.empty:
        print(f"❌ ERROR: no data to process!")
        print(f"   Lines read: {total_lines}")
        print(f"   Lines skipped: {skipped}")
        print(f"   Results: {len(results)}")
        exit(1)

    print(f"✓ Records processed: {total_lines} total, {skipped} skipped")
    print(f"✓ Result rows created: {len(results)}")

    # --- 2. Semantic scoring ---
    print(f"\n🧠 Computing semantic scores...")

    df['sem_score'] = 0.0

    # TP parfait (match exact) = 1.0
    tp_mask = (df['type'] == 'tp')
    df.loc[tp_mask, 'sem_score'] = 1.0

    # -------------------------------------------------------------
    # OPTIMIZATION: Batch encode all unique string values first
    # -------------------------------------------------------------
    print("⏳ Batch encoding all unique values...")
    unique_origs = set(df.loc[~tp_mask & (df['val_orig'].astype(str).str.strip() != ''), 'val_orig'].astype(str).str.strip().unique())
    unique_preds = set(df.loc[~tp_mask & (df['val_pred'].astype(str).str.strip() != ''), 'val_pred'].astype(str).str.strip().unique())
    all_unique_texts = list(unique_origs | unique_preds)
    
    # Pre-compute embeddings for all unique texts
    text_to_emb = {}
    if all_unique_texts:
        # Encode with batching (much faster than one-by-one)
        embeddings = semantic_model.encode(all_unique_texts, batch_size=256, convert_to_tensor=True)
        for text, emb in zip(all_unique_texts, embeddings):
            text_to_emb[text] = emb

    # FP/FN: compare the values of the SAME feature semantically
    unique_features = df['feature'].unique()
    for feature in tqdm(unique_features, desc="Semantic scoring", unit=" features"):
        feature_mask = (df['feature'] == feature)
        feature_data = df[feature_mask].copy()
        
        # Get the original and predicted values for this feature
        origins = feature_data[feature_data['val_orig'].astype(str).str.strip() != '']
        predictions = feature_data[feature_data['val_pred'].astype(str).str.strip() != '']
        
        if len(origins) > 0 and len(predictions) > 0:
            indices_orig = origins.index.tolist()
            indices_pred = predictions.index.tolist()
            
            # Compare the values
            for idx_o in indices_orig:
                if df.loc[idx_o, 'type'] == 'tp':
                    continue
                val_o = str(df.loc[idx_o, 'val_orig']).strip()
                emb1 = text_to_emb.get(val_o)
                if emb1 is None: continue
                
                for idx_p in indices_pred:
                    if df.loc[idx_p, 'type'] == 'tp':
                        continue
                    val_p = str(df.loc[idx_p, 'val_pred']).strip()
                    emb2 = text_to_emb.get(val_p)
                    if emb2 is None: continue
                    
                    sim = util.cos_sim(emb1, emb2).item()
                    
                    if df.loc[idx_o, 'type'] == 'fn':
                        if sim > df.loc[idx_o, 'sem_score']:
                            df.loc[idx_o, 'sem_score'] = sim
                    if df.loc[idx_p, 'type'] == 'fp':
                        if sim > df.loc[idx_p, 'sem_score']:
                            df.loc[idx_p, 'sem_score'] = sim

    df['sem_score'] = df['sem_score'].fillna(0.0)
    print(f"✓ Semantic scores computed")

    # --- 3. Summary per feature ---
    print(f"\n📊 Summarizing results by feature...")
    summary = []
    for feat, group in df.groupby('feature'):
        tp = len(group[group['type'] == 'tp'])
        fp = len(group[group['type'] == 'fp'])
        fn = len(group[group['type'] == 'fn'])
        
        # Precision and recall: computed only on the cases present (consistent with SEM)
        # Do not penalize rare features with missing_cases
        # The feature weight (count) is applied at the global level when needed
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        count = tp + fn
        count_pct = (count / total_lines) * 100
        avg_sem = group['sem_score'].mean()
        
        # Mean semantic score for this feature
        # Mean = sum of sem_scores / number of occurrences of the feature
        sem_with_missing = group['sem_score'].mean()
        
        summary.append({'feat': feat, 'prec': prec, 'rec': rec, 'f1': f1, 'sem': sem_with_missing, 'count': count, 'pct': count_pct, 'tp': tp, 'fp': fp, 'fn': fn})

    # Tri par occurence (count)
    summary_df = pd.DataFrame(summary).sort_values('count', ascending=False)

    # Custom feature order per mode
    if mode == "tmj":
        feature_order = [
            "patient_age", "headache_intensity", "tmj_pain_rating",
            "disc_displacement", "joint_arthritis_location", "jaw_function_score", "maximum_opening",
            "diet_score", "disability_rating", "tinnitus_present", "vertigo_present",
            "joint_pain_areas", "earache_present", "pain_aggravating_factors", "average_daily_pain_intensity",
            "airway_obstruction_present", "pain_onset_date", "appliance_history", "current_medications",
            "headache_location", "muscle_pain_location", "muscle_symptoms_present", "muscle_pain_score",
            "hearing_loss_present", "jaw_clicking", "headache_frequency", "sleep_disorder_type",
            "maximum_opening_without_pain", "neck_pain_present", "current_appliance", "onset_triggers",
            "physical_therapy_status", "adverse_reactions", "jaw_crepitus", "jaw_locking",
            "pain_relieving_factors", "back_pain_present", "sleep_apnea_diagnosed", "autoimmune_condition",
            "migraine_history", "previous_medications", "pain_frequency", "depression_present",
            "pain_duration", "fibromyalgia_present"
        ]
        total_features = 45
    else:
        # Orthodontie 140 
        feature_order = [
            "Upper Wire Size", "Upper Wire Material", "Lower Wire Size", "Lower Wire Material",
            "Changed Upper Arch Wire", "Changed Lower Arch Wire", "Oral Hygiene", "Ligature Method",
            "Right Molar Class", "Left Molar Class", "Left Canine Class", "Right Canine Class",
            "Elastic Pattern Left", "Elastic Pattern Right", "Compliance", "Elastic Type Left",
            "Elastic Type Right", "Overjet (mm)", "Overbite (mm)", "Recods Taken (x-rays; IOS; Photos, Facial scanning)",
            "Upper Arch Bends", "Lower Arch Bends", "Debonded Bracket", "Retainer Check",
            "Re-tie Appointment", "Appliance", "Emergency Type", "Prescription and Bracket Slot",
            "Bracket OR BAND Repositioning", "Retainer", "COS lower", "EMERGENCY",
            "Upper Bonding", "IPR", "Lower Bonding", "Open Spring",
            "Active Space Closure", "COS upper", "Posterior Bite Turbos", "Upper Debond",
            "Lower Debond", "Enameloplasty", "TADs", "Relapse",
            "TMJ symptoms", "Upper Banding", "TPA", "Referral",
            "Extractions", "Posterior Crossbite", "Lower Banding", "Closed Spring",
            "Anterior Bite Turbos", "Maxillary Expander", "Anterior Crossbite", "LLHA",
            "Mandibular Advancement Appliance", "NANCE", "Cantilever", "Tongue Crib",
            "FaceMask"
        ]
        total_features = 61

    # Map each feature (lowercase) to its index
    feature_order_lower = [f.lower() for f in feature_order]
    feature_index_map = {f.lower(): idx for idx, f in enumerate(feature_order)}

    # Append unlisted features at the end (in order of appearance)
    all_features = summary_df['feat'].tolist()
    for feat in all_features:
        if feat.lower() not in feature_index_map:
            feature_index_map[feat.lower()] = len(feature_index_map)

    # Sort summary_df by the custom order
    summary_df['sort_index'] = summary_df['feat'].str.lower().map(feature_index_map)
    summary_df = summary_df.sort_values('sort_index').drop('sort_index', axis=1)

    # Dictionary for quick feature lookup
    feat_dict = {}
    for idx, row in summary_df.iterrows():
        feat_dict[row['feat'].lower()] = row

    # Count the detected features (present in feat_dict)
    features_detected = len(feat_dict)

    # Build the FULL table with ALL features in the planned order (missing ones too)
    complete_features_list = []
    for order_feat in feature_order:
        order_feat_lower = order_feat.lower()
        if order_feat_lower in feat_dict:
            r = feat_dict[order_feat_lower]
            complete_features_list.append({
                'feat': r['feat'],
                'prec': r['prec'],
                'rec': r['rec'],
                'f1': r['f1'],
                'sem': r['sem'],
                'count': r['count'],
                'pct': r['pct'],
                'tp': r['tp'],
                'fp': r['fp'],
                'fn': r['fn']
            })
        else:
            # Missing feature: add it with "-"
            complete_features_list.append({
                'feat': order_feat,
                'prec': '-',
                'rec': '-',
                'f1': '-',
                'sem': '-',
                'count': '-',
                'pct': '-',
                'tp': '-',
                'fp': '-',
                'fn': '-'
            })

    # Convert to a DataFrame for saving
    summary_df_filtered = pd.DataFrame(complete_features_list)

    # Print the table title and headers
    print(f"\n📈 Results by Feature:\n")
    print(f"Total features: {features_detected} / {total_features}\n")
    print(f"{'FEATURE':<40} | {'PREC':<5} | {'REC':<5} | {'F1':<5} | {'SEM':<5} | {'COUNT':<5} | {'%':<5}")
    print("-" * 95)

    # Print every feature in feature_order, missing ones too
    for order_feat in feature_order:
        order_feat_lower = order_feat.lower()
        if order_feat_lower in feat_dict:
            r = feat_dict[order_feat_lower]
            print(f"{r['feat'][:40]:<40} | {r['prec']:.2f}  | {r['rec']:.2f}  | {r['f1']:.2f}  | {r['sem']:.2f}  | {int(r['count']):<5} | {r['pct']:.0f}%")
        else:
            # Missing feature: print dashes
            print(f"{order_feat[:40]:<40} | {'-':<5} | {'-':<5} | {'-':<5} | {'-':<5} | {'-':<5} | {'-':<5}")

    # --- 4. Global ---
    # Keep ONLY the detected features (drop the "-")
    detected_mask = summary_df_filtered['prec'] != '-'
    summary_df_detected = summary_df_filtered[detected_mask]

    # Plain mean of the scores of all detected features
    g_prec = summary_df_detected['prec'].mean() if len(summary_df_detected) > 0 else 0
    g_rec = summary_df_detected['rec'].mean() if len(summary_df_detected) > 0 else 0
    g_f1 = summary_df_detected['f1'].mean() if len(summary_df_detected) > 0 else 0

    # Global semantic score: mean SEM over the features present ("-" not counted)
    g_sem = summary_df_detected['sem'].mean() if len(summary_df_detected) > 0 else 0

    print("-" * 95)

    # --- 5. Plain and weighted totals ---
    # Plain total: every feature has the same weight
    print(f"{'TOTAL SIMPLE':<40} | {g_prec:.3f} | {g_rec:.3f} | {g_f1:.3f} | {g_sem:.3f}")

    # Weighted total: mean weighted by frequency (count)
    total_count = summary_df_detected['count'].sum()
    g_prec_weighted = (summary_df_detected['prec'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    g_rec_weighted = (summary_df_detected['rec'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    g_f1_weighted = (summary_df_detected['f1'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    g_sem_weighted = (summary_df_detected['sem'] * summary_df_detected['count']).sum() / total_count if total_count > 0 else 0
    print(f"{'TOTAL WEIGHTED':<40} | {g_prec_weighted:.3f} | {g_rec_weighted:.3f} | {g_f1_weighted:.3f} | {g_sem_weighted:.3f}")

    print("-" * 95)

    # --- 6. SAVE RESULTS ---
    print("\n💾 Saving results to Excel...")

    # Create base filename based on model type
    # NOTE: {mode}{table_number} (no separating underscore) matches the
    # naming convention used everywhere else in the pipeline, e.g.
    # training_data_{mode}{table_number}.jsonl, extraction_llama_{mode}{table_number}.jsonl.
    if model_type == "llama":
        base_filename = f"result_llama_{mode}{table_number}"
    elif model_type == "qwen":
        base_filename = f"result_qwen_{mode}{table_number}"
    else:
        base_filename = f"result_{mode}{table_number}"

    # Save XLSX (Excel format)
    xlsx_file = results_dir / f"{base_filename}.xlsx"
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        # Write decimals with a comma in the XLSX (French locale) - 3 decimals
        summary_df_xlsx = summary_df_filtered.copy()
        for col in ['prec', 'rec', 'f1', 'sem', 'count']:
            if col in summary_df_xlsx.columns:
                summary_df_xlsx[col] = summary_df_xlsx[col].apply(
                    lambda x: f"{float(x):.3f}".replace('.', ',') if isinstance(x, (int, float)) else x
                )
        # pct as a decimal fraction (5% -> 0,05)
        if 'pct' in summary_df_xlsx.columns:
            summary_df_xlsx['pct'] = summary_df_xlsx['pct'].apply(
                lambda x: f"{float(x)/100:.2f}".replace('.', ',') if isinstance(x, (int, float)) else x
            )
        
        # Sheet 1: Features
        summary_df_xlsx.to_excel(writer, sheet_name='Features', index=False)
        
        # Sheet 2: Global Metrics
        metrics_data = {
            'Mode': [mode],
            'F1': [f"{g_f1:.3f}".replace('.', ',')],
            'SEM': [f"{g_sem:.3f}".replace('.', ',')],
            'W F1': [f"{g_f1_weighted:.3f}".replace('.', ',')],
            'W SEM': [f"{g_sem_weighted:.3f}".replace('.', ',')]
        }
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name='Global Metrics', index=False)
        
        # Sheet 3: Summary Info
        summary_info = {
            'Parameter': ['Mode', 'Table Number', 'Total Samples', 'Features Detected', 'Features in Base List'],
            'Value': [mode, table_number, total_lines, features_detected, total_features]
        }
        summary_info_df = pd.DataFrame(summary_info)
        summary_info_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Format the workbook
        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"  ✓ XLSX: {xlsx_file}")
    print(f"\n✅ Results saved to: {results_dir}\n")

    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)
    print(f"✅ Evaluation complete!\n")


if __name__ == "__main__":
    main(table_number=5, mode="no_prompt", model_type="llama")