"""
Pipeline runner script to run the train -> run -> eval pipeline for every
(mode, table) combination listed in MODES x TABLES below, back to back,
sequentially reusing train_model.main / run_model.main / eval_model.main.

Usage:
    python run_pipeline.py                    # Run all (no_prompt, with_cot, without_cot) x (1,2,3,4)

Note: imports its steps as `model_scripts.train_model` etc (absolute
package imports, unlike the sibling scripts which add their own directory
to sys.path), so this must be run with the repo root on the Python path,
e.g. `python -m model_scripts.advanced.run_pipeline` from the repo root,
or with the repo root on PYTHONPATH - running `python run_pipeline.py`
directly from within model_scripts/advanced/ will fail to import
model_scripts.

Reads/writes: whatever train_model.py / run_model.py / eval_model.py each
read and write per (mode, table) - see their module docstrings. A failure
in any one step (caught per-config below) does not stop the remaining
configurations from running.
"""

from model_scripts.train_model import main as train_main
from model_scripts.run_model import main as run_main
from model_scripts.eval_model import main as eval_main

# ============================================================================
# CONFIGURATION - edit here to change what runs
# ============================================================================
MODES = ["no_prompt", "with_cot", "without_cot"]  # Modes to run
TABLES = [4]                              # Tables to run
PIPELINE_STEPS = ["train", "run", "eval"]         # Train → Run → Eval
# ============================================================================

def main():
    """
    Run train -> run -> eval, in order, for every (mode, table) pair in
    MODES x TABLES. Each step is called with model_type defaulted to "phi"
    (train_main/run_main/eval_main all default to "phi" when not passed).
    Errors in any single configuration are caught and logged so the loop
    continues to the next configuration instead of aborting the whole run.

    """
    total = len(MODES) * len(TABLES)
    count = 0
    
    print(f"\n{'='*80}")
    print(f"RUN_PIPELINE - Running {total} configurations")
    print(f"Modes: {', '.join(MODES)}")
    print(f"Tables: {', '.join(map(str, TABLES))}")
    print(f"{'='*80}\n")
    
    for mode in MODES:
        for table in TABLES:
            count += 1
            print(f"\n[{count}/{total}] {mode.upper()} - Table {table}")
            print("-" * 80)
            
            try:
                for step in PIPELINE_STEPS:
                    if step == "train":
                        print("  → Training...")
                        train_main(table_number=table, mode=mode)
                    elif step == "run":
                        print("  → Running inference...")
                        run_main(table_number=table, mode=mode, eval_set="test")
                    elif step == "eval":
                        print("  → Evaluating...")
                        eval_main(table_number=table, mode=mode)
                
                print(f"  ✓ Success\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
    
    print(f"\n{'='*80}")
    print("✓ ALL DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
