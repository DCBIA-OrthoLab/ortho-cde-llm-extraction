"""
Run inference with a fine-tuned model (trained by train_model.py) over a
patient-notes dataset and save the raw extraction predictions to disk.

Supports two backends:
  - precision="16bit": the merged safetensors model produced by
    train_model.py, loaded via Unsloth/transformers .generate().
  - precision="4bit": a quantized model-q4_k_m.gguf (must already exist next
    to the 16bit model, e.g. produced by an external llama.cpp conversion
    step) loaded via llama-cpp-python. Useful for measuring inference speed/
    memory under quantization.

Invocation:
    python run_model.py
        Runs main() with the hardcoded defaults in the __main__ block below
        (edit that call, or import `main` from another script, e.g.
        advanced/run_pipeline.py, to change table_number/mode/model_type/etc).

Env vars: sets HF_HUB_DOWNLOAD_TIMEOUT / HF_HUB_READ_TIMEOUT to 600s to
avoid Hugging Face Hub timeouts on slow connections.

Reads:
    model/{model_name}_{mode}{table_number}/ - the trained model directory
        (safetensors, or a model-q4_k_m.gguf inside it for 4bit).
    data/2_input_model/{mode}/training_data_{mode}{table_number}.jsonl
        (or an explicit `input_path`) - same input format as train_model.py.
        Re-applies grouped_shuffle_split with the same seed so eval_set
        ("train"/"validation"/"test") matches the split used at train time.

Writes:
    data/3_output_model/{mode}/extraction_{...}{mode}{table_number}[...].jsonl
        One JSON object per line: {"metadata", "original_note", "original"
        (ground-truth assistant message), "prediction" (raw model output,
        JSON-substring-extracted)}. The filename varies by model_type,
        full_dataset, and precision to avoid overwriting other runs.
"""

import os
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_READ_TIMEOUT"] = "600"

import json
import time
import statistics
from pathlib import Path
from datasets import load_dataset
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch
import sys

# Add model_scripts (this file's own directory) to sys.path so `from utils
# import ...` works whether this script is run directly or imported from
# elsewhere (e.g. advanced/run_pipeline.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import grouped_shuffle_split


def main(table_number: int, mode: str, eval_set: str = "test", model_type: str = "phi", full_dataset: bool = False, precision: str = "16bit", input_path: str = None, preloaded_dataset=None):
    """
    Main function for running the model and generating predictions.

    Args:
        table_number: Table number to use
        mode: Mode (e.g., "no_prompt", "tmj")
        eval_set: Which set to evaluate ("train", "validation", "test") - ignored if full_dataset or dataset is given
        model_type: Model type ("phi" or "llama")
        full_dataset: If True, perform inference on the entire dataset loaded from
            input_path/json_path (ignores eval_set split)
        precision: "16bit" (merged safetensors via Unsloth) or "4bit" (model-q4_k_m.gguf via llama.cpp)
        input_path: Optional explicit path to the jsonl, overriding the default
            data/2_input_model/{mode}/training_data_{mode}{table_number}.jsonl
        dataset: Optional pre-loaded, unformatted HF Dataset (with 'messages' and
            'metadata') to use as the generation set directly - e.g. a fixed, shared
            test set. Used as-is, no grouped_shuffle_split, no file read. Used by the
            learning_curve experiment to pass in-memory subsets without a jsonl per
            point.
    """
    print("-" * 95)
    print(f" {mode}, Table {table_number}, Model: {model_type}, Precision: {precision}")
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

    # Path Configuration
    script_folder = Path(__file__).resolve().parent
    project_root = script_folder.parent

    model_dir = project_root / "model" / f"{config['model_name'].split('/')[-1]}_{mode}{table_number}"
    model_path = str(model_dir)
    gguf_path = str(model_dir / "model-q4_k_m.gguf")
    if preloaded_dataset is None:
        json_path = input_path or str(project_root / "data" / "2_input_model" / f"{mode}" / f"training_data_{mode}{table_number}.jsonl")
    if model_type == "llama":
        output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_llama_{mode}{table_number}.jsonl")
    elif model_type == "qwen":
        output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_qwen_{mode}{table_number}.jsonl")
    else:
        output_path = str(project_root / "data" / "3_output_model" / f"{mode}" / f"extraction_{mode}{table_number}_verif.jsonl")

    # Modify output filename if using full dataset
    if full_dataset:
        output_path = output_path.replace(".jsonl", "_full_dataset.jsonl")

    # Tag the output filename so 4bit runs never overwrite 16bit results
    if precision == "4bit":
        output_path = output_path.replace(".jsonl", "_4bit.jsonl")

    # -------------------------------------------------------------------------
    # 1. Load Model
    # -------------------------------------------------------------------------
    print(f"Load Model ({precision})...")
    if precision == "4bit":
        from llama_cpp import Llama
        from transformers import AutoTokenizer

        if not Path(gguf_path).exists():
            raise FileNotFoundError(f"4bit GGUF not found: {gguf_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        llm = Llama(
            model_path=gguf_path,
            n_gpu_layers=-1,
            n_ctx=max_seq_length,
            verbose=False,
        )
        model = None
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(model)

    # -------------------------------------------------------------------------
    # 2. Load Data (Same split as training)
    # -------------------------------------------------------------------------
    print("Load Data...")

    if preloaded_dataset is not None:
        generation_dataset = preloaded_dataset
        print(f"Evaluating on pre-loaded dataset (used as-is): {len(generation_dataset)} notes")
    else:
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

        if precision == "4bit":
            text_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

            t0 = time.perf_counter()
            # max_tokens=647 comfortably covers the JSON extraction schema's
            # worst-case output length; temperature=0.0 for deterministic,
            # reproducible extractions. stop tokens cover all three model
            # families' end-of-turn markers (Llama/Qwen/Phi) since this
            # script is shared across model_type values.
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
        else:
            input_ids = tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            # do_sample=False + temperature=0.0 for deterministic greedy
            # decoding (reproducible extractions); max_new_tokens=647 mirrors
            # the 4bit branch's budget above.
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=647,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            sample_gen_time = t1 - t0
            gen_times.append(sample_gen_time)

            # sample current GPU memory allocated (GB)
            if torch.cuda.is_available():
                try:
                    mem_gb = torch.cuda.memory_allocated() / 1024**3
                except Exception:
                    mem_gb = 0.0
                memory_samples.append(mem_gb)
            # (no system RAM sampling)

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                out_len = outputs.shape[-1] if hasattr(outputs, 'shape') else (outputs[0].numel() if hasattr(outputs[0], 'numel') else len(outputs[0]))
            except Exception:
                out_len = 0
            try:
                if isinstance(input_ids, dict):
                    first = next(iter(input_ids.values()))
                else:
                    first = input_ids
                in_len = first.shape[-1]
            except Exception:
                in_len = 0
            generated_tokens = max(0, out_len - in_len)

        # The model may prefix its answer with extra text/markdown before the
        # JSON extraction object, so take everything from the first "{" as
        # the candidate JSON payload (the saved "prediction" is this raw
        # substring, not a re-serialized/validated object - eval_model.py
        # does its own json.loads downstream).
        start_index = generated_text.find("{")
        if start_index != -1:
            prediction_json = generated_text[start_index:]
        else:
            prediction_json = ""

        # valid/invalid_json_count are tallied here for a possible future
        # validity-rate report but are not currently printed or saved.
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
    # precision="16bit" -> safetensors via Unsloth | precision="4bit" -> model-q4_k_m.gguf via llama.cpp
    main(table_number="5", mode="no_prompt", eval_set="test", model_type="phi", full_dataset=False, precision="16bit")

