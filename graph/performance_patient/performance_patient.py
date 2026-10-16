"""Plot F1 and semantic-similarity scores as a function of training-set size.

Unlike the other scripts in graph/, this one does NOT read a data file: the
F1/semantic scores per patient-count checkpoint (14, 18, 22, 44, 140, 180
patients) are hardcoded below, transcribed from prior evaluation runs.
Produces a 2x2 grid of line plots (Macro F1, Weighted F1, Macro Semantic,
Weighted Semantic vs. number of training patients), with a reference dashed
line for the Gemini 3 Pro dry-run baseline on the weighted subplots.

Output: ``graph/performance_patient/performance_patient.png``.
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

# Repository root (this file lives in graph/<figure>/), so paths work on any machine
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# F1 data
data_f1 = {
    'Categorie': ['Macro F1', 'Weighted F1'],
    '14 Patients': [0.291, 0.462],
    '18 Patients': [0.369, 0.549],
    '22 Patients': [0.385, 0.57],
    '44 Patients': [0.518, 0.73],
    '140 Patients': [0.606, 0.75],
    '180 Patients': [0.588, 0.75]
}

# Semantic data (order: [Macro, Weighted])
data_sem = {
    'Categorie': ['Macro Semantic Similarity', 'Weighted Semantic Similarity'],
    '14 Patients': [0.580, 0.841],
    '18 Patients': [0.672, 0.867],
    '22 Patients': [0.728, 0.889],
    '44 Patients': [0.763, 0.944],
    '140 Patients': [0.950, 0.984],
    '180 Patients': [0.960, 0.99]
}

df_f1 = pd.DataFrame(data_f1)
df_sem = pd.DataFrame(data_sem)

patient_counts = {
    '14 Patients': 14,
    '18 Patients': 18,
    '22 Patients': 22,
    '44 Patients': 44,
    '140 Patients': 140,
    '180 Patients': 180
}

# Create 4 plots (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Macro F1
row = df_f1[df_f1['Categorie'] == 'Macro F1'].iloc[0]
x_vals = [patient_counts[col] for col in df_f1.columns if col != 'Categorie']
y_vals = [row[col] for col in df_f1.columns if col != 'Categorie']
# Column order in the dict above is not numerically sorted by patient count,
# so re-sort the (x, y) pairs before plotting a line to avoid a zig-zag line.
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[0, 0].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[0, 0].text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
axes[0, 0].set_title('F1 - Macro (Unweighted)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylabel('F1 Score', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks([14, 18, 22, 44, 140, 180])
axes[0, 0].set_ylim(0.2, 1.0)

# Weighted F1
row = df_f1[df_f1['Categorie'] == 'Weighted F1'].iloc[0]
y_vals = [row[col] for col in df_f1.columns if col != 'Categorie']
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[0, 1].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#ff7f0e')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[0, 1].text(x, y + 0.003, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
# Baseline from a separate one-off Gemini 3 Pro dry-run evaluation (not derived from data above).
axes[0, 1].axhline(y=0.485, color='gray', linestyle='--', linewidth=2, label='Gemini 3 Pro (Dry Run)')
axes[0, 1].set_title('F1 - Weighted', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks([14, 18, 22, 44, 140, 180])
axes[0, 1].set_ylim(0.4, 1.0)
axes[0, 1].legend(loc='lower right')

# Macro Semantic Similarity
row = df_sem[df_sem['Categorie'] == 'Macro Semantic Similarity'].iloc[0]
x_vals = [patient_counts[col] for col in df_sem.columns if col != 'Categorie']
y_vals = [row[col] for col in df_sem.columns if col != 'Categorie']
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[1, 0].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#2ca02c')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[1, 0].text(x, y + 0.01, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
axes[1, 0].set_title('Semantic - Macro (Unweighted)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Number of Patients', fontsize=12)
axes[1, 0].set_ylabel('Semantic Score', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xticks([14, 18, 22, 44, 140, 180])
axes[1, 0].set_ylim(0.5, 1.0)

# Weighted Semantic Similarity
row = df_sem[df_sem['Categorie'] == 'Weighted Semantic Similarity'].iloc[0]
y_vals = [row[col] for col in df_sem.columns if col != 'Categorie']
sorted_points = sorted(zip(x_vals, y_vals))
x_sorted, y_sorted = zip(*sorted_points)
axes[1, 1].plot(x_sorted, y_sorted, marker='o', linewidth=2.5, markersize=8, color='#d62728')
for i, (x, y) in enumerate(zip(x_sorted, y_sorted)):
    axes[1, 1].text(x, y + 0.003, f'{y:.3f}', ha='center', va='bottom', fontsize=10)
axes[1, 1].axhline(y=0.668, color='gray', linestyle='--', linewidth=2, label='Gemini 3 Pro (Dry Run)')
axes[1, 1].set_title('Semantic - Weighted', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Number of Patients', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_xticks([14, 18, 22, 44, 140, 180])
axes[1, 1].set_ylim(0.6, 1.0)
axes[1, 1].legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, 'graph/performance_patient/performance_patient.png'), dpi=300, bbox_inches='tight')
print("✓ Graph saved: performance_patient.png")
plt.show()