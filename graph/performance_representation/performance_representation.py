"""Plot per-feature F1/semantic score against feature frequency.

Reads the per-feature result table ``result_llama_no_prompt6_all.xlsx``
(under ``data/5_results/no_prompt/``), which has one row per feature with
``feat`` (name), ``pct`` (frequency share), ``f1``, and ``sem`` columns.
Produces two scatter plots (F1 vs frequency, Semantic score vs frequency)
with a linear trend line each; the 10 most frequent features are
highlighted with distinct colors (rest shown in gray) and a shared legend
below both plots. Also prints summary statistics (means overall, for
frequent features >50%, and rare features <10%) to stdout.

Output: ``graph/performance_representation/performance_vs_frequency.png``.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Repository root (this file lives in graph/<figure>/), so paths work on any machine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load data from Excel file
df_raw = pd.read_excel(os.path.join(PROJECT_ROOT, 'data/5_results/no_prompt/result_llama_no_prompt6_all.xlsx'))

# Helper function to convert French formatted numbers to float
def parse_value(val):
    """Convert a French-formatted numeric cell (comma decimal, spaces) to
    float. Returns None for missing values or the '-' placeholder."""
    if pd.isna(val) or val == '-':
        return None
    if isinstance(val, str):
        return float(val.replace(',', '.').replace(' ', ''))
    return float(val)

# Prepare lists to build the DataFrame
filtered_data = []
for _, row in df_raw.iterrows():
    f1_val = parse_value(row['f1'])
    if f1_val is not None:
        filtered_data.append({
            'Feature': row['feat'],
            'Count_Pct': parse_value(row['pct']) * 100 if row['pct'] is not None else 0,
            'F1': f1_val,
            'SEM': parse_value(row['sem'])
        })

df = pd.DataFrame(filtered_data)

# Find the 10 most frequent features
top_10_df = df.nlargest(10, 'Count_Pct').reset_index(drop=True)
top_10_indices = df.nlargest(10, 'Count_Pct').index

# Clearly distinct colors (full spectrum, no near-duplicates)
colors_palette = ['#FF0000', "#FF6600", "#740000", '#00CC00', '#0000FF',
                  '#4B0082', '#FF1493', '#00CED1', '#FFD700', "#00310F"]

# Build the color list for each point
colors_f1 = []
colors_sem = []
for idx in df.index:
    if idx in top_10_indices:
        # Find the position of the point in the top 10
        position = list(top_10_indices).index(idx)
        colors_f1.append(colors_palette[position])
        colors_sem.append(colors_palette[position])
    else:
        colors_f1.append('#CCCCCC')  # Grey for the others
        colors_sem.append('#CCCCCC')

# Create figure with 2 subplots - taller to fit the legend
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: F1 vs Count %
ax1 = axes[0]
ax1.scatter(df['Count_Pct'], df['F1'], alpha=0.7, s=100, c=colors_f1, edgecolors='black', linewidth=0.5)
# Add trend line
z1 = np.polyfit(df['Count_Pct'], df['F1'], 1)
p1 = np.poly1d(z1)
x_smooth = np.linspace(df['Count_Pct'].min(), df['Count_Pct'].max(), 100)
ax1.plot(x_smooth, p1(x_smooth), "darkgray", linewidth=2, label='F1 Trend')
ax1.set_xlabel('Count % (Feature Frequency)', fontsize=12, fontweight='bold')
ax1.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax1.set_title('F1 Score Evolution Based on Feature Frequency', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-2, 105)
ax1.set_ylim(-0.05, 1.05)

# Plot 2: SEM vs Count %
ax2 = axes[1]
ax2.scatter(df['Count_Pct'], df['SEM'], alpha=0.7, s=100, c=colors_sem, edgecolors='black', linewidth=0.5)
# Add trend line
z2 = np.polyfit(df['Count_Pct'], df['SEM'], 1)
p2 = np.poly1d(z2)
ax2.plot(x_smooth, p2(x_smooth), "darkgray", linewidth=2, label='SEM Trend')
ax2.set_xlabel('Count % (Feature Frequency)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Semantic Score', fontsize=12, fontweight='bold')
ax2.set_title('Semantic Score Evolution Based on Feature Frequency', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-2, 105)
ax2.set_ylim(-0.05, 1.05)

# Build the shared legend below the plots
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors_palette[i], edgecolor='black', 
                         label=top_10_df.iloc[i]['Feature']) 
                   for i in range(len(top_10_df))]
legend_elements.append(Patch(facecolor='#CCCCCC', edgecolor='black', label='Other Features'))

# Place the legend below both plots
fig.subplots_adjust(bottom=0.25)
fig.legend(handles=legend_elements, fontsize=11, loc='lower center', ncol=6, 
           bbox_to_anchor=(0.5, 0.01), frameon=True, edgecolor='black', fancybox=True)

plt.savefig(os.path.join(PROJECT_ROOT, 'graph/performance_representation/performance_vs_frequency.png'), dpi=300)
print("✓ Graph saved: performance_vs_frequency.png")

# Statistics
print("\n📊 Global Statistics:")
print(f"  - Number of features: {len(df)}")
print(f"  - Average F1 Score: {df['F1'].mean():.3f}")
print(f"  - Average Semantic Score: {df['SEM'].mean():.3f}")
print(f"  - Average Count %: {df['Count_Pct'].mean():.1f}%")
print(f"\n  - F1 Score of frequent features (>50%): {df[df['Count_Pct'] > 50]['F1'].mean():.3f}")
print(f"  - SEM of frequent features (>50%): {df[df['Count_Pct'] > 50]['SEM'].mean():.3f}")
print(f"\n  - F1 Score of rare features (<10%): {df[df['Count_Pct'] < 10]['F1'].mean():.3f}")
print(f"  - SEM of rare features (<10%): {df[df['Count_Pct'] < 10]['SEM'].mean():.3f}")

plt.show()
