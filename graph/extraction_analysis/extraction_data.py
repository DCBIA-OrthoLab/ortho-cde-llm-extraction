"""Summarize extraction match rates and counts per feature (Figures 1 & 2).

Reads the same row-level extraction comparison table as
advanced_analysis.py: ``extraction_features_no_prompt6_all.xlsx``
(under ``data/4_model_table/no_prompt/``), with one row per
(Patient_ID, Feature) extraction and ``Manual``/``Model`` value columns.
For each feature it computes the exact-match rate and how many times the
feature was present in the manual vs. model output, then plots the top 20
most frequent features:

- Figure 1: horizontal bar chart of exact-match rate per feature.
- Figure 2: horizontal grouped bar chart of Manual vs Model extraction
  counts per feature.

Outputs: ``graph/extraction_analysis/exact_match_rate.png`` and
``graph/extraction_analysis/extraction_counts.png``.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Repository root (this file lives in graph/<figure>/), so paths work on any machine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Configuration ---
FILE_PATH = os.path.join(PROJECT_ROOT, 'data/4_model_table/no_prompt/extraction_features_no_prompt6_all.xlsx')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'graph/extraction_analysis')

print(f"Loading data from: {FILE_PATH}")
df = pd.read_excel(FILE_PATH)

def clean_val(val):
    """Normalize a Manual/Model cell for comparison: lowercase, strip
    whitespace and the 'mm' unit suffix, map missing values to ""."""
    if pd.isna(val):
        return ""
    # Convert to string, lower, strip whitespaces, remove "mm" for comparisons
    val_str = str(val).lower().strip().replace('mm', '').strip()
    return val_str

# Clean the Manual and Model columns for a fairer comparison
df['Manual_clean'] = df['Manual'].apply(clean_val)
df['Model_clean'] = df['Model'].apply(clean_val)

# Exact Match Calculation
df['Exact_Match'] = df['Manual_clean'] == df['Model_clean']

# Calculate Metrics per Feature
features = df['Feature'].unique()
metrics = []

for feat in features:
    sub_df = df[df['Feature'] == feat]
    
    # We consider an extraction 'present' if it's not empty
    manual_count = (sub_df['Manual_clean'] != "").sum()
    model_count = (sub_df['Model_clean'] != "").sum()
    total_comparisons = len(sub_df)
    
    exact_matches = sub_df['Exact_Match'].sum()
    match_rate = exact_matches / total_comparisons if total_comparisons > 0 else 0
    
    metrics.append({
        'Feature': feat,
        'Total': total_comparisons,
        'Manual_Count': manual_count,
        'Model_Count': model_count,
        'Match_Rate': match_rate
    })

df_metrics = pd.DataFrame(metrics)

# Sort by total comparisons to get the most frequent features
df_metrics = df_metrics.sort_values(by='Total', ascending=False)
top_features = df_metrics.head(20) # Take top 20 for readability

# --- Plotting ---
plt.style.use('default')

# Figure 1: Exact Match Rate
fig1, ax1 = plt.subplots(figsize=(12, 8), facecolor='white')
bars = ax1.barh(top_features['Feature'][::-1], top_features['Match_Rate'][::-1], color='#4682b4', edgecolor='black', linewidth=1)

ax1.set_title('Exact Match Rate by Feature (Top 20 Frequent)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Exact Match Rate', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 1.1)
ax1.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
ax1.set_axisbelow(True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar in bars:
    w = bar.get_width()
    ax1.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.1%}', ha='left', va='center', fontsize=10)

fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, 'exact_match_rate.png')
fig1.savefig(out1, dpi=300)
print(f"✓ Saved: {out1}")

# Figure 2: Number of Extractions (Manual vs Model)
fig2, ax2 = plt.subplots(figsize=(14, 8), facecolor='white')
y = np.arange(len(top_features))
height = 0.35

ax2.barh(y - height/2, top_features['Manual_Count'][::-1], height, label='Manual (Ground Truth)', color='#1f4e79', edgecolor='black')
ax2.barh(y + height/2, top_features['Model_Count'][::-1], height, label='Model Prediction', color='#9bc2e6', edgecolor='black')

ax2.set_title('Number of Extractions per Feature: Manual vs Model', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Count', fontsize=12, fontweight='bold')
ax2.set_yticks(y)
ax2.set_yticklabels(top_features['Feature'][::-1])
ax2.grid(axis='x', linestyle='--', alpha=0.6, zorder=0)
ax2.set_axisbelow(True)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend()

fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, 'extraction_counts.png')
fig2.savefig(out2, dpi=300)
print(f"✓ Saved: {out2}")
