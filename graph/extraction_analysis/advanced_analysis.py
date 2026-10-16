"""Analyze per-instance extraction errors (Figures 3 & 4).

Reads the row-level extraction comparison table
``extraction_features_no_prompt6_all.xlsx`` (under
``data/4_model_table/no_prompt/``), which is expected to have one row per
(Patient_ID, Feature) extraction with ``Manual`` (ground truth) and
``Model`` (prediction) value columns. Each row is classified into an error
type (Exact Match / Value Mismatch / Missed / Hallucinated) and this is
used to produce two figures:

- Figure 3: stacked horizontal bar chart of error-type counts for the
  top 15 most frequent features.
- Figure 4: histogram of per-patient exact-match rate, with mean/median
  markers.

Outputs: ``graph/extraction_analysis/error_typology_stacked.png`` and
``graph/extraction_analysis/patient_accuracy_distribution.png``.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Repository root (this file lives in graph/<figure>/), so paths work on any machine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
FILE_PATH = os.path.join(PROJECT_ROOT, 'data/4_model_table/no_prompt/extraction_features_no_prompt6_all.xlsx')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'graph/extraction_analysis')

print(f"Loading data from: {FILE_PATH}")
df = pd.read_excel(FILE_PATH)

def clean_val(val):
    """Normalize a Manual/Model cell for comparison: lowercase, strip whitespace
    and the 'mm' unit suffix, and map missing/empty markers to "" so both
    sides compare equal when the feature was genuinely absent."""
    if pd.isna(val) or val == 'None' or val == 'NaN':
        return ""
    return str(val).lower().strip().replace('mm', '').strip()

df['Manual_clean'] = df['Manual'].apply(clean_val)
df['Model_clean'] = df['Model'].apply(clean_val)

# --- Define Error Types ---
def classify_extraction(row):
    """Classify one row's Manual vs Model value into an error-type bucket
    used for the stacked bar chart (Figure 3)."""
    man = row['Manual_clean']
    mod = row['Model_clean']

    if man != "" and mod != "":
        if man == mod:
            return 'Exact Match'
        else:
            return 'Value Mismatch'
    elif man != "" and mod == "":
        return 'Missed (False Negative)'
    elif man == "" and mod != "":
        return 'Hallucinated (False Positive)'
    else:
        return 'Both Empty'

df['Error_Type'] = df.apply(classify_extraction, axis=1)
# Filter out Both Empty if any snuck in
df = df[df['Error_Type'] != 'Both Empty']

# --- Figure 3: Error Typology by Feature ---
# Get top 15 features by total occurrences
top_feats = df['Feature'].value_counts().head(15).index

error_counts = df[df['Feature'].isin(top_feats)].groupby(['Feature', 'Error_Type']).size().unstack(fill_value=0)

# Sort by 'Exact Match'
error_counts['Total'] = error_counts.sum(axis=1)
error_counts = error_counts.sort_values(by='Exact Match', ascending=True)
error_counts = error_counts.drop(columns=['Total'])

# Desired column order and colors for scientific paper
col_order = ['Exact Match', 'Value Mismatch', 'Missed (False Negative)', 'Hallucinated (False Positive)']
colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd']

# Ensure all columns exist
for col in col_order:
    if col not in error_counts:
        error_counts[col] = 0
error_counts = error_counts[col_order]

fig3, ax3 = plt.subplots(figsize=(14, 8), facecolor='white')
error_counts.plot(kind='barh', stacked=True, color=colors, ax=ax3, edgecolor='black', linewidth=0.5, alpha=0.9)

ax3.set_title('Extraction Error Typology per Feature (Top 15)', fontsize=15, fontweight='bold', pad=15)
ax3.set_xlabel('Number of Extracted Instances', fontsize=12, fontweight='bold')
ax3.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax3.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.legend(title='Prediction Status', loc='lower right', framealpha=1)

fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, 'error_typology_stacked.png')
fig3.savefig(out3, dpi=300)
print(f"✓ Saved: {out3}")

# --- Figure 4: Patient-Level Consistency ---
# Calculate exact match rate per Patient
patient_stats = df.groupby('Patient_ID').apply(
    lambda x: (x['Error_Type'] == 'Exact Match').sum() / len(x) * 100
).reset_index(name='Match_Rate_Pct')

fig4, ax4 = plt.subplots(figsize=(10, 6), facecolor='white')
ax4.hist(patient_stats['Match_Rate_Pct'], bins=20, color='#1f4e79', edgecolor='black', alpha=0.8)

mean_rate = patient_stats['Match_Rate_Pct'].mean()
median_rate = patient_stats['Match_Rate_Pct'].median()

ax4.axvline(mean_rate, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_rate:.1f}%')
ax4.axvline(median_rate, color='orange', linestyle='dotted', linewidth=2, label=f'Median: {median_rate:.1f}%')

ax4.set_title('Distribution of Accuracy Across Patients', fontsize=15, fontweight='bold', pad=15)
ax4.set_xlabel('Patient-Level Exact Match Rate (%)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Number of Patients', fontsize=12, fontweight='bold')
ax4.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.legend()

fig4.tight_layout()
out4 = os.path.join(OUTPUT_DIR, 'patient_accuracy_distribution.png')
fig4.savefig(out4, dpi=300)
print(f"✓ Saved: {out4}")

