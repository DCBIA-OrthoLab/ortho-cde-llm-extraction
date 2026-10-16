"""
Hyperparameter search for the Phi-3.5-mini LoRA fine-tune, using Optuna -
a Two Phase Strategy:
    Phase 1: Optimize HIGH PRIORITY params (LR, batch_size, epochs)
    Phase 2: Optimize MEDIUM PRIORITY params (grad_accum, lora_r, warmup,
             scheduler) - requires manually copying Phase 1's winning
             values into the `objective()` Phase 2 branch first (see the
             "CHANGE THIS" comments below).

Invocation:
    python optimize_hyperparams.py
        Runs main() with the hardcoded table_number/mode/phase at the
        bottom of this file - edit those to change what gets optimized.
    Disables W&B logging for trials (wandb.disabled = True); Optuna's own
    console output (and the saved JSON below) is the source of truth.

Reads:
    data/2_input_model/{mode}/training_data_{mode}{table_number}.jsonl
        Same input format as train_model.py. Loaded and tokenized once in
        HyperparameterOptimizer.__init__ (not per-trial) to save time.

Writes:
    model/optimization_phase{phase}_{mode}_{table_number}.json
        Best hyperparameters, best validation loss, and full per-trial
        history for the run.
    model/temp_trial/
        Scratch training output dir for each trial, deleted after every
        trial (trials are not intended to be resumed/kept).
"""

import json
import optuna
from optuna.pruners import MedianPruner
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import wandb
import shutil
import sys

# Add model_scripts (the parent of this advanced/ folder) to the path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import grouped_shuffle_split

class HyperparameterOptimizer:
    """
    Runs an Optuna study over one training-config "phase" for a fixed
    (table_number, mode). Loads and tokenizes the dataset once up front
    (see _prepare_dataset), then each Optuna trial (objective) loads a
    fresh base model + LoRA adapters, trains briefly, and reports back the
    best validation loss reached as the value Optuna tries to minimize.
    """
    def __init__(self, table_number: int, mode: str, phase: int = 1, n_trials: int = None):
        """
        Initialize the optimizer
        
        Args:
            table_number: Table number (1-4)
            mode: Training mode
            phase: 1 for HIGH PRIORITY, 2 for MEDIUM PRIORITY
            n_trials: Number of optimization trials (auto if None)
        """
        self.table_number = table_number
        self.mode = mode
        self.phase = phase
        
        # Auto-set number of trials based on phase (currently the same value
        # for both phases - kept as separate branches for easy per-phase
        # tuning later).
        if n_trials is None:
            self.n_trials = 12 if phase == 1 else 12
        else:
            self.n_trials = n_trials

        # Setup paths. This file lives in model_scripts/advanced/, so the
        # repo root (where data/ and model/ live) is two levels up.
        script_folder = Path(__file__).resolve().parent
        self.project_root = script_folder.parent.parent
        self.json_path = str(
            self.project_root / "data" / "2_input_model" / f"{mode}" /
            f"training_data_{mode}{table_number}.jsonl"
        )
        self.output_dir = str(self.project_root / "model")
        
        # Determine max_seq_length
        self.max_seq_length = 6144 if mode == "tmj" else 2048
        
        # Default fixed params (from train_model.py)
        self.fixed_params = {
            "lora_dropout": 0.0,
            "bias": "none",
            "use_gradient_checkpointing": "unsloth",
            "random_state": 3407
        }
        
        print("-" * 95)
        print(f"🔬 Hyperparameter Optimization: {mode}, Table {table_number}")
        print(f"📊 Phase {phase} - {'HIGH PRIORITY (LR, batch, epochs)' if phase == 1 else 'MEDIUM PRIORITY (grad_accum, lora_r, warmup, scheduler)'}")
        print(f"🔄 Number of trials: {self.n_trials}")
        print("-" * 95)
        
        # ⚡ OPTIMIZATION: Load and tokenize dataset once in __init__
        print("\n⚡ Pre-loading and tokenizing dataset (one-time cost)...")
        self._prepare_dataset()
        print("✅ Dataset ready!\n")
    
    def _prepare_dataset(self):
        """
        Load, tokenize and split dataset ONCE.
        Store tokenized splits in self.dataset_split for reuse across trials.
        This saves 20-30% time by avoiding repeated tokenization.
        """
        # Load model tokenizer (temporary, just for tokenization)
        _, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Phi-3.5-mini-instruct",
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        
        # Load dataset
        dataset = load_dataset("json", data_files=self.json_path, split="train")
        print(f"  Loaded {len(dataset)} samples")
        
        # Tokenize once
        def format_single_patient(patient_record):
            conversation = patient_record["messages"]
            formatted_texts = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False,
                add_generation_prompt=False
            )
            return {"text": formatted_texts}
        
        print(f"  Tokenizing...")
        dataset = dataset.map(format_single_patient, desc="Tokenizing")
        
        # Split 70/15/15 using Grouped Shuffle Split
        split_result = grouped_shuffle_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        
        self.dataset_split = {
            'train': split_result['train'],
            'validation': split_result['validation'],
            'test': split_result['test']
        }
        
        # Clean up tokenizer
        del tokenizer
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization
        Returns: validation loss (lower is better)
        """
        
        # =====================================================================
        # 1. SUGGEST HYPERPARAMETERS (based on phase)
        # =====================================================================
        
        if self.phase == 1:
            # ═══════════════════════════════════════════════════════════════
            # PHASE 1: HIGH PRIORITY PARAMETERS
            # ═══════════════════════════════════════════════════════════════
            
            # Learning rate - logarithmic scale (most important!)
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            
            # Batch size - discrete options
            batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
            
            # Number of epochs
            num_epochs = trial.suggest_categorical("num_epochs", [3, 5])
            
            # Fixed from best results
            grad_accum = 2  # Optimized
            lora_r = 32     # Optimized
            warmup_ratio = 0.148  # Optimized
            lr_scheduler = "cosine"  # Optimized
            weight_decay = 0.02  # Optimized
            
        else:  # phase == 2
            # ═══════════════════════════════════════════════════════════════
            # PHASE 2: MEDIUM PRIORITY PARAMETERS
            # Load best params from Phase 1 first!
            # ═══════════════════════════════════════════════════════════════
            
            # Fixed from Phase 1 (use your best results)
            lr = 2e-4  # CHANGE THIS to your best Phase 1 result
            batch_size = 4  # CHANGE THIS to your best Phase 1 result
            num_epochs = 3  # CHANGE THIS to your best Phase 1 result
            
            # Optimize these in Phase 2
            grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8])
            lora_r = trial.suggest_categorical("lora_r", [16, 32, 64])
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.02, 0.15)
            lr_scheduler = trial.suggest_categorical(
                "lr_scheduler_type", 
                ["linear", "cosine"]
            )
            weight_decay = trial.suggest_float("weight_decay", 0.0, 0.05)
        
        # ⚡ Calculate lora_alpha ONCE (applies to both phases)
        lora_alpha = lora_r * 2
        
        # Log trial info
        print(f"\n{'='*70}")
        print(f"Trial {trial.number + 1}/{self.n_trials} (Phase {self.phase})")
        print(f"{'='*70}")
        print(f"📌 Learning Rate:     {lr:.2e}")
        print(f"📌 Batch Size:        {batch_size}")
        print(f"📌 Epochs:            {num_epochs}")
        if self.phase == 2:
            print(f"🔧 Grad Accum:        {grad_accum}")
            print(f"🔧 LoRA r:            {lora_r}")
            print(f"🔧 Warmup Ratio:      {warmup_ratio:.3f}")
            print(f"🔧 LR Scheduler:      {lr_scheduler}")
            print(f"🔧 Weight Decay:      {weight_decay:.3f}")
        
        try:
            # =====================================================================
            # 2. LOAD MODEL
            # =====================================================================
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Phi-3.5-mini-instruct",
                max_seq_length=self.max_seq_length,
                dtype=None,
                load_in_4bit=False,
            )
            
            # =====================================================================
            # 3. APPLY LORA
            # =====================================================================
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=lora_alpha,  # ⚡ Use the variable (Phase 2: varies with r, Phase 1: fixed at 128)
                lora_dropout=0.0,  # Fixed
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3407
            )
            
            # =====================================================================
            # 4. LOAD AND PREPARE DATA
            # =====================================================================
            # ⚡ OPTIMIZATION: Use pre-loaded tokenized dataset instead of reloading
            dataset_split = self.dataset_split
            
            # =====================================================================
            # 5. TRAIN
            # =====================================================================
            response_template = "<|assistant|>\n"
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer
            )
            
            # Calculate total steps for warmup
            total_steps = (
                len(dataset_split['train']) // (batch_size * grad_accum) * num_epochs
            )
            warmup_steps = int(total_steps * warmup_ratio)
            
            # Disable wandb for trials
            wandb.disabled = True
            
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset_split["train"],
                eval_dataset=dataset_split["validation"],
                dataset_text_field="text",
                max_seq_length=self.max_seq_length,
                data_collator=collator,
                dataset_num_proc=8,
                packing=False,
                
                args=TrainingArguments(
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    gradient_accumulation_steps=grad_accum,
                    learning_rate=lr,
                    fp16=False,
                    bf16=True,
                    optim="adamw_8bit",
                    
                    num_train_epochs=num_epochs,
                    warmup_ratio=warmup_ratio,
                    
                    logging_steps=max(1, total_steps // 20),
                    report_to="none",
                    
                    eval_strategy="steps",
                    eval_steps=max(1, total_steps // 10),
                    save_strategy="no",
                    
                    output_dir=str(Path(self.output_dir) / "temp_trial"),
                    overwrite_output_dir=True,
                    weight_decay=weight_decay,
                    lr_scheduler_type=lr_scheduler,
                    seed=3407,
                ),
            )
            
            # Train and get final validation loss
            trainer_stats = trainer.train()
            
            # Get best validation loss from trainer.state
            # The best eval loss is saved in trainer.state during training
            best_val_loss = trainer.state.best_metric if trainer.state.best_metric is not None else float('inf')
            
            # If best_metric is not the eval_loss, get it from log history
            if best_val_loss == float('inf') or best_val_loss is None:
                # Get the last eval_loss from training history
                for log_entry in reversed(trainer.state.log_history):
                    if 'eval_loss' in log_entry:
                        best_val_loss = log_entry['eval_loss']
                        break
                else:
                    best_val_loss = float('inf')
            
            print(f"\n✅ Best Validation Loss: {best_val_loss:.4f}")
            
            # Cleanup
            del model, tokenizer, trainer
            torch.cuda.empty_cache()
            
            # Clean temp directory
            temp_dir = Path(self.output_dir) / "temp_trial"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            
            return best_val_loss
        
        except Exception as e:
            print(f"❌ Trial failed: {str(e)}")
            torch.cuda.empty_cache()
            return float('inf')
    
    def optimize(self):
        """Run the optimization"""
        
        # Create study
        study = optuna.create_study(
            direction="minimize",  # Lower validation loss is better
            pruner=MedianPruner(n_startup_trials=3)
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # =====================================================================
        # RESULTS
        # =====================================================================
        print("\n" + "=" * 95)
        print("🎯 OPTIMIZATION RESULTS - PHASE " + str(self.phase))
        print("=" * 95)
        
        # Best trial
        best_trial = study.best_trial
        print(f"\n✨ Best Trial: #{best_trial.number}")
        print(f"📊 Best Validation Loss: {best_trial.value:.4f}")
        print(f"\n🏆 Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key:.<40} {value}")
        
        # Top 5 trials
        print(f"\n📈 Top 5 Best Trials:")
        sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
        for i, trial in enumerate(sorted_trials[:5], 1):
            if trial.value is not None:
                print(f"    {i}. Trial #{trial.number}: Val Loss = {trial.value:.4f}")
        
        # Save results to JSON
        results_path = Path(self.output_dir) / f"optimization_phase{self.phase}_{self.mode}_{self.table_number}.json"
        results = {
            "phase": self.phase,
            "mode": self.mode,
            "table_number": self.table_number,
            "best_trial_number": best_trial.number,
            "best_validation_loss": float(best_trial.value),
            "best_hyperparameters": best_trial.params,
            "all_trials": [
                {
                    "trial_number": t.number,
                    "validation_loss": float(t.value) if t.value is not None else None,
                    "hyperparameters": t.params,
                    "state": str(t.state)
                }
                for t in study.trials
            ]
        }
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {results_path}")
        print("=" * 95)
        
        if self.phase == 1:
            print("\n📝 NEXT STEPS:")
            print("   1. Copy 'best_hyperparameters' from Phase 1 results")
            print("   2. Update Phase 2 script with these values:")
            print(f"      lr = {best_trial.params.get('learning_rate', 5e-4)}")
            print(f"      batch_size = {best_trial.params.get('batch_size', 4)}")
            print(f"      num_epochs = {best_trial.params.get('num_epochs', 5)}")
            print("   3. Run Phase 2 optimization")
        else:
            print("\n📝 NEXT STEPS:")
            print("   1. Copy 'best_hyperparameters' from Phase 2 results")
            print("   2. Update train_model.py with ALL best parameters")
            print("   3. Run final training: python train_model.py")
            print("   4. Evaluate: python eval_model.py")
        
        return best_trial.params


def main():
    """Main entry point"""
    table_number = 4
    mode = "no_prompt"
    phase = 1  # Change to 2 for Phase 2 optimization
    
    # For Phase 2, make sure to update the fixed params with Phase 1 best results!
    if phase == 2:
        print("\n⚠️  IMPORTANT FOR PHASE 2:")
        print("   Update the fixed params in optimize_hyperparams.py:")
        print("   - lr = best_lr from Phase 1")
        print("   - batch_size = best_batch from Phase 1")
        print("   - num_epochs = best_epochs from Phase 1")
    
    optimizer = HyperparameterOptimizer(
        table_number=table_number,
        mode=mode,
        phase=phase,
        n_trials=None  # Auto-set based on phase
    )
    
    best_params = optimizer.optimize()
    
    print("\n" + "=" * 95)
    print("🎉 OPTIMIZATION COMPLETE!")
    print("=" * 95)


if __name__ == "__main__":
    main()
