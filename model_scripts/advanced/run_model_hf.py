"""
Variant of run_model.py that downloads its model from a public Hugging Face
Hub repo instead of a local model/ directory, for reproducing inference on
a machine that never ran train_model.py locally. The HF repo only ships a
q4_k_m GGUF (no 16bit safetensors), so this script only supports 4bit
inference via llama-cpp-python.

Invocation:
    python run_model_hf.py
        Runs main() with the hardcoded defaults in the __main__ block below.

Env vars: sets HF_HUB_DOWNLOAD_TIMEOUT / HF_HUB_READ_TIMEOUT to 600s to
avoid Hugging Face Hub timeouts on slow connections.

Reads:
    HF_REPO_ID / HF_GGUF_FILENAME (below) - downloaded via hf_hub_download
        (cached locally by huggingface_hub after the first run).
    data/2_input_model/{mode}/training_data_{mode}{table_number}.jsonl -
        same input format as train_model.py/run_model.py; re-applies
        grouped_shuffle_split with the same seed so eval_set matches the
        split used at train time.

Writes:
    data/3_output_model/{mode}/extraction_{...}{mode}{table_number}_hf_ortho_4bit.jsonl
        Same per-line record format as run_model.py's output.
"""

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

import json
import time
import statistics
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
import sys

# Add model_scripts (the parent of this advanced/ folder) to the path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import grouped_shuffle_split

# Same as run_model.py, but the model is downloaded from the HF Hub repo
# below instead of a local folder. That repo only ships a q4_k_m GGUF
# (no 16bit weights), so this script only supports 4bit inference.
HF_REPO_ID = "DCBIA-OrthoLab/Meta-Llama-3.1-8B-Instruct-Ortho"
HF_GGUF_FILENAME = "model-q4_k_m.gguf"


def main(table_number: int, mode: str, eval_set: str = "test", model_type: str = "phi", full_dataset: bool = False):
    """
    Main function for running the model and generating predictions.

    Args:
        table_number: Table number to use
        mode: Mode (e.g., "no_prompt", "tmj")
        eval_set: Which set to evaluate ("train", "validation", "test") - ignored if full_dataset=True
        model_type: Model type ("phi" or "llama")
        full_dataset: If True, perform inference on the entire dataset (ignores eval_set split)
    """
    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}, Precision: 4bit (HF Hub: {HF_REPO_ID})")
    print("-" * 95)

    # Determine max_seq_length based on mode
    if mode == "tmj":
        max_seq_length = 6144
    else:
        max_seq_length = 2048

    # Model configuration
    model_configs = {
        "phi": {"model_name": "unsloth/Phi-3.5-mini-instruct",},
        "llama": {"model_name": "unsloth/Meta-Llama-3.1-8B-Instruct",},
        "qwen": {"model_name": "unsloth/Qwen2.5-7B-Instruct",}
    }
    config = model_configs[model_type]

    print(f"Run on {eval_set} set")

    # Path Configuration. This file lives in model_scripts/advanced/, so the
    # repo root (where data/ lives) is two levels up.
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent.parent

    json_path = str(project_root / "data" / "2_input_model" / f"{mode}" / f"training_data_{mode}{table_number}.jsonl")
    if model_type == "llama":
        output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_llama_{mode}{table_number}.jsonl")
    elif model_type == "qwen":
        output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_qwen_{mode}{table_number}.jsonl")
    else:
        output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_{mode}{table_number}_verif.jsonl")

    # Modify output filename if using full dataset
    if full_dataset:
        output_path = output_path.replace(".jsonl", "_full_dataset.jsonl")

    # Tag the output filename so HF Ortho runs never overwrite local-model results
    output_path = output_path.replace(".jsonl", "_hf_ortho_4bit.jsonl")

    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    print(f"Download GGUF from HF Hub ({HF_REPO_ID}/{HF_GGUF_FILENAME})...")
    gguf_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_GGUF_FILENAME)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    llm = Llama(
        model_path=gguf_path,
        n_gpu_layers=-1,
        n_ctx=max_seq_length,
        verbose=False,
    )

    # -------------------------------------------------------------------------
    # 2. Load Data (Same split as training)
    # -------------------------------------------------------------------------
    print("Load Data...")
    dataset = load_dataset("json", data_files=json_path, split="train")

    if full_dataset:
        generation_dataset = dataset
        print(f"Evaluating on FULL DATASET: {len(generation_dataset)} notes")
    else:
        dataset = grouped_shuffle_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
        generation_dataset = dataset[eval_set]
        print(f"Evaluating on {eval_set.upper()} set: {len(generation_dataset)} notes")
        print(f"  Train: {len(dataset['train'])} | Validation: {len(dataset['validation'])} | Test: {len(dataset['test'])}")

    # -------------------------------------------------------------------------
    # 3. Generation
    # -------------------------------------------------------------------------
    print("Generation...")
    result = []
    valid_json_count = 0
    invalid_json_count = 0
    gen_times = []

    # Metrics tracking
    total_generated_tokens = 0
    memory_samples = []
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_t0 = time.perf_counter()

    for patient_record in tqdm(generation_dataset):
        prompt = patient_record["messages"][:-1]
        truth = patient_record["messages"][-1]["content"]

        metadata = patient_record.get("metadata", {})
        patient_id = metadata.get("patient_id", "Unknown")
        note_date = metadata.get("note_date", "Unknown")
        note_month = metadata.get("note_month", "Unknown")
        if hasattr(note_date, 'isoformat'):
            note_date = note_date.isoformat()
        if hasattr(note_month, 'isoformat'):
            note_month = note_month.isoformat()

        text_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        t0 = time.perf_counter()
        # max_tokens=647 comfortably covers the JSON extraction schema's
        # worst-case output length; temperature=0.0 for deterministic,
        # reproducible extractions. stop tokens cover all three model
        # families' end-of-turn markers (Llama/Qwen/Phi), matching
        # run_model.py's 4bit branch.
        response = llm(
            text_prompt,
            max_tokens=647,
            temperature=0.0,
            echo=False,
            stop=["<|eot_id|>", "<|im_end|>", "<|end|>"],
        )
        t1 = time.perf_counter()
        sample_gen_time = t1 - t0
        gen_times.append(sample_gen_time)

        generated_tokens = response["usage"]["completion_tokens"]
        in_len = response["usage"]["prompt_tokens"]
        out_len = in_len + generated_tokens

        generated_text = response["choices"][0]["text"]

        # The model may prefix its answer with extra text/markdown before
        # the JSON extraction object, so take everything from the first "{"
        # as the candidate JSON payload (same heuristic as run_model.py).
        start_index = generated_text.find("{")
        if start_index != -1:
            prediction_json = generated_text[start_index:]
        else:
            prediction_json = ""

        try:
            json.loads(prediction_json)
            valid_json_count += 1
        except (json.JSONDecodeError, ValueError):
            invalid_json_count += 1

        result.append({
            "metadata": {"patient_id": patient_id, "note_date": note_date, "note_month": note_month},
            "original_note": prompt[-1]["content"],
            "original": truth,
            "prediction": prediction_json
        })

        total_generated_tokens += generated_tokens

    # finalize wall time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_t1 = time.perf_counter()
    wall_total = wall_t1 - wall_t0

    # Compute and print only the requested inference metrics
    total_tokens = total_generated_tokens
    total_gen_time = sum(gen_times)
    latency_per_note = statistics.mean(gen_times) if gen_times else 0

    # Compute samples-based throughput and total samples
    total_samples = len(gen_times)
    inference_throughput = total_samples / total_gen_time if total_gen_time > 0 else 0

    print(f"Total Inference (all samples): {total_samples}")
    print(f"Inference Throughput (samples/s): {inference_throughput:.2f}")
    print(f"Latency per Note (avg generation time): {latency_per_note:.3f} s")
    wall_total_min = wall_total / 60.0 if wall_total is not None else None
    if wall_total_min is not None:
        print(f"Full run time (wall-clock): {wall_total:.3f} s ({wall_total_min:.3f} min)")
    else:
        print("Full run time (wall-clock): N/A")

    # VRAM stats (GB)
    peak_vram_gb = None
    avg_vram_gb = None
    if torch.cuda.is_available():
        try:
            peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        except Exception:
            peak_vram_gb = max(memory_samples) if memory_samples else None
        avg_vram_gb = statistics.mean(memory_samples) if memory_samples else None

    if peak_vram_gb is not None:
        print(f"VRAM usage peak (GB): {peak_vram_gb:.2f}")
    else:
        print("VRAM usage peak (GB): N/A")
    if avg_vram_gb is not None:
        print(f"VRAM usage avg (GB): {avg_vram_gb:.2f}")
    else:
        print("VRAM usage avg (GB): N/A")

    # -------------------------------------------------------------------------
    # 4. Save
    # -------------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        for item in result:
            json_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved as {output_path}")

    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}")
    print("-" * 95)


if __name__ == "__main__":
    # To run inference on the FULL DATASET, use full_dataset=True
    # Model loaded from the HF Hub (DCBIA-OrthoLab/Meta-Llama-3.1-8B-Instruct-Ortho), 4bit only
    main(table_number="5", mode="no_prompt", eval_set="test", model_type="llama", full_dataset=False)
