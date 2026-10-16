"""
Fine-tune a base LLM (Phi-3.5-mini, Llama-3.1-8B or Qwen2.5-7B) with
Unsloth + LoRA to extract structured clinical fields from patient notes.

Invocation:
    python train_model.py
        Runs main() with the hardcoded defaults in the __main__ block below
        (edit that call, or import `main` from another script, e.g.
        advanced/run_pipeline.py, to change table_number/mode/model_type).

Reads:
    data/2_input_model/{mode}/training_data_{mode}{table_number}.jsonl
        (or an explicit `input_path`) - a JSONL file where each line is a
        record with a "messages" field (chat-format conversation: system/
        user/assistant turns) and a "metadata" field containing at least
        "patient_id" (required by grouped_shuffle_split for patient-level
        splitting).

Writes:
    model/{model_name}_{mode}{table_number}/
        The merged (LoRA + base weights) 16-bit model and tokenizer, saved
        via Unsloth's save_pretrained_merged. Intermediate training
        checkpoints under model/checkpoint-* are deleted after the run.

Requires a Weights & Biases login (wandb.login()) since training metrics are
reported to W&B (report_to="wandb").
"""

from datasets import load_dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
import wandb
from pathlib import Path
import shutil
import sys
import time

# Add model_scripts (this file's own directory) to sys.path so `from utils
# import ...` works whether this script is run directly or imported from
# elsewhere (e.g. advanced/run_pipeline.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import grouped_shuffle_split


def main(table_number: int, mode: str, model_type: str = "phi", input_path: str = None, train_dataset=None, val_dataset=None):
    """
    Main function for training the model.

    Args:
        table_number: Table number (1-4)
        mode: Training mode (no_prompt, with_cot, without_cot, tmj, dry_run)
        model_type: Model to use ("phi" or "llama")
        input_path: Optional explicit path to the training jsonl, overriding the
            default data/2_input_model/{mode}/training_data_{mode}{table_number}.jsonl
        train_dataset: Optional pre-loaded, unformatted HF Dataset (with 'messages'
            and 'metadata') to use as train data directly, instead of reading from
            input_path/json_path. Used by the learning_curve experiment to pass
            in-memory patient subsets without writing a jsonl per point.
        val_dataset: Optional pre-loaded, unformatted HF Dataset to use as the fixed
            validation set directly. If provided (with train_dataset), no
            grouped_shuffle_split is performed - train_dataset and val_dataset are
            used as-is.
    """
    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)

    # TMJ notes/schema are considerably longer than the orthodontic ones, so
    # TMJ mode needs a much larger context window to avoid truncating notes.
    if mode == "tmj":
        max_seq_length = 6144
    else:
        max_seq_length = 2048

    # Model configuration: base checkpoint + LoRA target_modules per model
    # family. target_modules lists which linear layers get LoRA adapters;
    # it differs by architecture (e.g. Llama-3.1 here omits "gate_proj").
    model_configs = {
        "phi": {
            "model_name": "unsloth/Phi-3.5-mini-instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "llama": {
            "model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
        },
        "qwen": {
            "model_name": "unsloth/Qwen2.5-7B-Instruct",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
            ],
        }
    }
    
    config = model_configs[model_type]

    # -------------------------------------------------------------------------
    # 1. Path Configuration
    # -------------------------------------------------------------------------
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    if train_dataset is None:
        json_path = input_path or str(project_root/"data"/"2_input_model"/f"{mode}"/f"training_data_{mode}{table_number}.jsonl")
    output_dir = str(project_root / "model")
    run_name = f"{config['model_name'].split('/')[-1]}_{mode}{table_number}"

    # -------------------------------------------------------------------------
    # 2. Load Model
    # -------------------------------------------------------------------------
    print("Load Model...")
    # dtype=None lets Unsloth auto-select the best dtype for the GPU;
    # load_in_4bit=False trains in full 16-bit precision (not QLoRA) since
    # the merged model is later saved as "merged_16bit".
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = False,
    )

    # -------------------------------------------------------------------------
    # 3. LoRA Configuration
    # -------------------------------------------------------------------------
    # use_gradient_checkpointing="unsloth" enables Unsloth's optimized
    # checkpointing (lower VRAM than the standard HF implementation).
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = config["target_modules"],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407
    )

    # -------------------------------------------------------------------------
    # 4. Data 
    # -------------------------------------------------------------------------
    print("Load Data...")
    dataset = train_dataset if train_dataset is not None else load_dataset("json", data_files=json_path, split="train")
    print("\n")
    print(f"Load {len(dataset)} cases")
    print("\n")

    def format_single_patient(patient_record):
        """
        Render one record's chat-format `messages` (system/user/assistant
        turns) into a single training string using the model's own chat
        template. add_generation_prompt=False because the target completion
        (assistant turn) is already included - this is for training, not
        generation.
        """
        conversation = patient_record["messages"]
        formatted_texts = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        return { "text" : formatted_texts }

    dataset = dataset.map(format_single_patient)

    if val_dataset is not None:
        # Fixed, pre-split validation set (learning_curve experiment): train_dataset/
        # input_path is used entirely as train data, val_dataset entirely as validation.
        val_formatted = val_dataset.map(format_single_patient)
        print(f"Using pre-split data: {len(dataset)} train notes, {len(val_formatted)} validation notes")
        dataset = {'train': dataset, 'validation': val_formatted, 'test': None}
    else:
        # Split train/val/test using Grouped Shuffle Split (patient-level integrity)
        split_result = grouped_shuffle_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        dataset = {
            'train': split_result['train'],
            'validation': split_result['validation'],
            'test': split_result['test']
        }

    # -------------------------------------------------------------------------
    # 5. Training
    # -------------------------------------------------------------------------
    # The completion-only collator masks the loss on everything before the
    # assistant turn, so it needs the exact literal token sequence each
    # model's chat template uses to introduce that turn.
    if model_type == "phi":
        response_template = "<|assistant|>\n"
    elif model_type == "qwen":
        response_template = "<|im_start|>assistant\n"
    else:  # llama
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    # Create data collator only for non-TMJ modes (TMJ has issues with custom collator)
    data_collator = None
    if mode != "tmj":
        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer
        )

    print("Start Fine Tuning...")
    wandb.login()
    trainer = SFTTrainer( 
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = dataset["validation"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = data_collator,
        dataset_num_proc = 12,
        packing = False,
        
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            per_device_eval_batch_size = 4,
            gradient_accumulation_steps = 2,
            learning_rate = 2e-4,
            fp16 = False,
            bf16 = True,
            optim = "adamw_8bit",

            num_train_epochs = 1, #3
            warmup_ratio = 0.148,
            
            logging_steps = 1,
            report_to = "wandb",
            run_name = run_name,

            group_by_length = True,

            eval_strategy = "steps",
            eval_steps = 5,
            save_strategy ="steps",
            save_total_limit = 3,

            load_best_model_at_end = True,
            metric_for_best_model = "eval_loss",

            output_dir = output_dir,
            overwrite_output_dir = True,  
            weight_decay = 0.02,
            lr_scheduler_type = "cosine",
            seed = 3407,
        ),
    )

    # Training 
    # Warm GPU sync then time the full training run (includes all epochs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_t0 = time.perf_counter()
    trainer_stats = trainer.train()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_t1 = time.perf_counter()

    total_train_time = train_t1 - train_t0
    # Calculate approximate time per note (across all epochs)
    num_train_notes = len(dataset['train'])
    num_epochs = getattr(trainer.args, 'num_train_epochs', 1)
    processed_notes = num_train_notes * num_epochs if num_train_notes > 0 else 0

    # Throughput (samples/sec)
    throughput = (processed_notes / total_train_time) if (processed_notes > 0 and total_train_time > 0) else None

    # Convergence info (epochs and global steps when available)
    global_steps = None
    try:
        global_steps = getattr(trainer.state, 'global_step', None)
    except Exception:
        global_steps = None

    # VRAM / peak GPU memory
    vram_str = "Unavailable"
    devices = None
    gpu_name = None
    try:
        if torch.cuda.is_available():
            devices = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            peak = None
            try:
                peak = torch.cuda.max_memory_reserved()
            except Exception:
                try:
                    peak = torch.cuda.max_memory_allocated()
                except Exception:
                    peak = None
            if peak:
                vram_str = f"{peak/1024**3:.2f} GB VRAM on {devices}x {gpu_name}"
            else:
                vram_str = f"GPU present ({devices}x {gpu_name}), peak memory unavailable"
        else:
            vram_str = "CPU only"
    except Exception:
        vram_str = "Unavailable"

    # Training time string (display in minutes)
    minutes = total_train_time / 60.0
    if torch.cuda.is_available() and devices is not None and gpu_name is not None:
        train_time_str = f"{minutes:.2f} minutes on {devices}x {gpu_name}"
    else:
        train_time_str = f"{minutes:.2f} minutes on CPU"

    # Print only the four requested metrics
    if throughput is not None:
        throughput_str = f"{throughput:.3f} samples/s"
    else:
        throughput_str = "unavailable"

    if global_steps is not None:
        convergence_str = f"Epochs: {num_epochs}, Global steps: {global_steps}"
    else:
        convergence_str = f"Epochs: {num_epochs}"

    print("-" * 95)
    print(f"Training Time : {train_time_str}")
    print(f"Throughput : {throughput_str}")
    print(f"Convergence Speed : {convergence_str}")
    print(f"VRAM Usage : {vram_str}")
    print("-" * 95)

    # -------------------------------------------------------------------------
    # 6. Save 
    # -------------------------------------------------------------------------
    print("Save Merge Model (Lora + Base) ...")
    model_save_path = Path(output_dir)/run_name
    model_save_path.mkdir(parents=True, exist_ok=True)

    # Save LoRA + base model
    model.save_pretrained_merged(
        str(model_save_path),
        tokenizer,
        save_method = "merged_16bit"
    )

    # Clean Checkpoints 
    for folder in Path(output_dir).glob("checkpoint-*"):
        try:
            shutil.rmtree(folder)
        except:
            pass
    print("Success")

    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)


if __name__ == "__main__":
    # Edit these arguments (or call main() from another script, e.g.
    # advanced/run_pipeline.py) to change which table/mode/model gets
    # trained. model_type also accepts "qwen" and "llama".
    main(table_number=5, mode="no_prompt", model_type="phi")
