"""
Shared utility functions used across the model_scripts pipeline.

Not a CLI entry point - imported by train_model.py, run_model.py,
advanced/optimize_hyperparams.py and advanced/run_model_hf.py, each of which
inserts this file's parent directory onto sys.path so `from utils import
grouped_shuffle_split` resolves regardless of the caller's working directory.

Currently exposes a single function, grouped_shuffle_split, which performs a
patient-level (grouped) train/validation/test split so that no single
patient's notes are spread across multiple splits.
"""

from collections import defaultdict
import random


def grouped_shuffle_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Grouped Shuffle Split: Split dataset by patient (Group) to prevent data leakage.
    
    Medical ML best practice: Each patient's notes stay together in ONE split only 
    (Train, Validation, or Test). Uses greedy bin packing to balance the splits by 
    total notes.
    
    Reference: "Group Shuffle Split" - scikit-learn convention for grouped data
    Algorithm: "Greedy Bin Packing" - heuristic for balanced distribution
    
    Args:
        dataset: HuggingFace Dataset with 'metadata' containing 'patient_id'
        train_ratio: Proportion of patients for training (default 0.7 = 70%)
        val_ratio: Proportion of patients for validation (default 0.15 = 15%)
        test_ratio: Proportion of patients for testing (default 0.15 = 15%)
        seed: Random seed for reproducibility (default 42)
    
    Returns:
        dict with keys 'train', 'validation', 'test' containing split datasets
    
    Example:
        >>> splits = grouped_shuffle_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        >>> train_set = splits['train']
        >>> val_set = splits['validation']
        >>> test_set = splits['test']
    """
    random.seed(seed)
    
    # Step 1: Group indices by patient
    patient_indices = defaultdict(list)
    for idx, record in enumerate(dataset):
        patient_id = record['metadata']['patient_id']
        patient_indices[patient_id].append(idx)
    
    # Step 2: Get unique patients and their note counts
    patients = list(patient_indices.keys())
    patient_notes_count = {p: len(patient_indices[p]) for p in patients}
    total_notes = len(dataset)
    
    # Step 3: Shuffle patients (randomize order)
    random.shuffle(patients)
    
    # Step 4: Greedy bin packing - allocate patients to splits
    # IMPORTANT: Fill small buckets FIRST (Test, Val) then Train gets the rest
    # This ensures Test and Val are balanced before Train absorbs leftovers
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    train_notes = 0
    val_notes = 0
    test_notes = 0
    
    train_target = int(total_notes * train_ratio)
    val_target = int(total_notes * val_ratio)
    test_target = int(total_notes * test_ratio)
    
    # Greedy strategy: For each patient, assign to the bucket "most underutilized"
    # This minimizes deviation from target percentages
    for patient in patients:
        patient_note_count = patient_notes_count[patient]
        indices = patient_indices[patient]
        
        # Calculate utilization ratio for each bucket (current / target)
        train_util = train_notes / max(train_target, 1)
        val_util = val_notes / max(val_target, 1)
        test_util = test_notes / max(test_target, 1)
        
        # Assign to the bucket that is most underutilized (lowest ratio)
        # Priority: Test/Val first (small buckets), Train last (absorbs rest)
        if test_util < 1.0 and test_util <= min(train_util, val_util):
            test_indices.extend(indices)
            test_notes += patient_note_count
        elif val_util < 1.0 and val_util <= min(train_util, test_util):
            val_indices.extend(indices)
            val_notes += patient_note_count
        elif train_util < 1.0:
            train_indices.extend(indices)
            train_notes += patient_note_count
        else:
            # Fallback: if all are "full", add to least full one
            if test_util <= val_util and test_util <= train_util:
                test_indices.extend(indices)
                test_notes += patient_note_count
            elif val_util <= train_util:
                val_indices.extend(indices)
                val_notes += patient_note_count
            else:
                train_indices.extend(indices)
                train_notes += patient_note_count
    
    # Step 5: Create split datasets
    train_set = dataset.select(train_indices)
    val_set = dataset.select(val_indices)
    test_set = dataset.select(test_indices)
    
    # Print statistics for verification
    unique_train_patients = len(set(dataset[i]['metadata']['patient_id'] for i in train_indices))
    unique_val_patients = len(set(dataset[i]['metadata']['patient_id'] for i in val_indices))
    unique_test_patients = len(set(dataset[i]['metadata']['patient_id'] for i in test_indices))
    
    print(f"\n✅ Grouped Shuffle Split (Greedy Bin Packing):")
    print(f"   Train: {len(train_set):4d} notes from {unique_train_patients:2d} patients ({100*len(train_set)/total_notes:5.1f}%)")
    print(f"   Val:   {len(val_set):4d} notes from {unique_val_patients:2d} patients ({100*len(val_set)/total_notes:5.1f}%)")
    print(f"   Test:  {len(test_set):4d} notes from {unique_test_patients:2d} patients ({100*len(test_set)/total_notes:5.1f}%)")
    print(f"   Total: {unique_train_patients + unique_val_patients + unique_test_patients} unique patients (no overlap)")
    print()
    
    return {
        'train': train_set,
        'validation': val_set,
        'test': test_set
    }
