# Medical-LLM-FineTuning

Fine-tuning and evaluation pipeline for extracting structured clinical findings
(e.g. TMJ / dental exam features) from free-text patient notes using open LLMs
(Llama 3.1, Qwen 2.5, Phi-3.5) with [Unsloth](https://github.com/unslothai/unsloth) + LoRA.

The pipeline turns manually-annotated clinical spreadsheets into LLM training
data, fine-tunes a model to extract the same structured fields from raw notes,
runs the model on held-out patients, and scores its output against the human
annotations (exact-match + semantic similarity).

## Authors

This pipeline was written by **[Paul-dumont](https://github.com/Paul-dumont)**,
who developed it from December 2025 to July 2026, starting from an initial
version by **[PyGuy2](https://github.com/PyGuy2)**.

**[Jules-GP](https://github.com/Jules-GP)** prepared this release (fixed the
paths and bugs a fresh clone runs into, pinned the dependencies, tested the
pipeline end to end on synthetic notes) and is listed by GitHub as the
contributor of its commits, but did not write the pipeline.

## Repository layout

```
data/                   Pipeline data, one stage per numbered folder
  1_manual_table/        Manually annotated patient spreadsheets (xlsm) - source of truth
  2_input_model/         Training data built from 1_manual_table (JSONL, chat format)
  3_output_model/        Raw model extractions from run_model.py (JSONL)
  4_model_table/         3_output_model flattened into a manual-vs-model comparison table (xlsx)
  5_results/             Scored results from eval_model.py (xlsx)

data_scripts/           Build & inspect the JSONL training data (see below)
model_scripts/          Train, run, and evaluate models (the core pipeline)
  advanced/               Orchestration, hyperparameter search, HF Hub inference variant
graph/                   Plots/figures produced from data/4_model_table and data/5_results

model/                   Trained model checkpoints (gitignored, generated locally)
wandb/                   Weights & Biases run logs (gitignored, generated locally)
```

`data/`, `model/`, and `wandb/` are gitignored. No data and no trained model
ship with this repository: the annotated clinical notes the models were
developed on are patient data and stay private. Bring your own data in the
format below; every script creates the folders it writes to.

## Input data format

`data_scripts/generate_json.py` reads one workbook,
`data/1_manual_table/patients_table<N>.xlsm` (set `table_number` at the top of
the script):

- One sheet per patient, plus a **last sheet that is skipped** (a legend or
  summary sheet).
- Columns, in this order: an index column, `Patient ID`, `Note Date`,
  `Note Month`, then any number of feature columns. An `Annotated` column, if
  present, is dropped.
- One column must be named `Raw Note Text`: the free-text clinical note. Rows
  without a note are dropped.
- Every other column is a feature to extract. An empty cell means "not
  mentioned"; a checkbox / `1` becomes `true`; any other value is kept as is.

The patient ID is only used to split train / validation / test by patient, so
notes from the same patient never land in two splits.

To use a trained model on new notes, write them as a JSONL file in the format
`generate_json.py` produces (one record per note, with `metadata.patient_id`
and a chat-format `messages` list whose last turn is the reference
extraction) and pass it to `run_model.py` with `input_path=` and
`full_dataset=True`.

## Pipeline

```
data_scripts/generate_json.py          data/1_manual_table -> data/2_input_model
        |
        v
model_scripts/train_model.py           data/2_input_model -> model/<name>
        |
        v
model_scripts/run_model.py             model/<name> -> data/3_output_model
        |
        v
model_scripts/eval_model.py            data/3_output_model -> data/5_results
        |
        v
graph/*/*.py                           data/4_model_table, data/5_results -> figures
```

`model_scripts/advanced/run_pipeline.py` chains train → run → eval for a set
of (mode, table) configs. `model_scripts/utils.py` holds the shared
patient-level train/val/test splitting logic used across the pipeline.

### data_scripts/

Builds and sanity-checks the JSONL training data from the manual annotation
workbooks (`data/1_manual_table/patients_table*.xlsm`):

| Script | Purpose |
|---|---|
| `generate_json.py` | Builds JSONL training sets in several prompting modes (with_cot, without_cot, dry_run, no_prompt, unknow) from a manual annotation workbook. |
| `generate_table.py` | Flattens a model-output JSONL (manual vs. predicted) into a long-format Excel table for manual review. |
| `print_json.py` | Pretty-prints one training example from a JSONL file for spot-checking. |
| `print_numbers.py` | Prints quick counts (patients, records, annotated features) for a training JSONL. |
| `print_representation.py` | Prints per-feature frequency/coverage stats across a training JSONL. |
| `print_tokens.py` | Tokenizes each record and reports avg/min/max token length (context-window sanity check). |
| `print_values.py` | Reports the value distribution for every feature. |

These scripts take their target table number / prompting mode as hardcoded
variables at the top of the file — edit them before running.

### model_scripts/

| Script | Purpose |
|---|---|
| `train_model.py` | Fine-tunes a base model (Phi-3.5-mini / Llama-3.1-8B / Qwen2.5-7B) with Unsloth + LoRA on `data/2_input_model/<mode>/...jsonl`; saves the merged model to `model/`. |
| `run_model.py` | Loads a trained model from `model/` and runs inference over train/val/test/full-dataset, writing predictions to `data/3_output_model/<mode>/...jsonl`. |
| `eval_model.py` | Scores `run_model.py` output against manual annotations (exact + semantic-similarity metrics) and writes an Excel report to `data/5_results/<mode>/`. |
| `utils.py` | Shared `grouped_shuffle_split()` for patient-level dataset splitting. |
| `advanced/run_pipeline.py` | Runs train → run → eval in sequence for a list of (mode, table) configs. |
| `advanced/optimize_hyperparams.py` | Optuna-based two-phase hyperparameter search (learning rate, batch size, etc.) for the Phi model. Standalone, not part of the main chain. |
| `advanced/run_model_hf.py` | Variant of `run_model.py` that pulls a quantized GGUF model from a public Hugging Face Hub repo instead of a local `model/` checkpoint. |

### graph/

Produces the figures used to report results, reading from
`data/4_model_table` and `data/5_results`:

| Script | Output |
|---|---|
| `compare_splits/compare_splits.py` | `splits_comparison.png` — per-feature F1/SEM averaged across All/Validation/Test splits. |
| `extraction_analysis/advanced_analysis.py` | `error_typology_stacked.png`, `patient_accuracy_distribution.png` — error-type breakdown and per-patient accuracy. |
| `extraction_analysis/extraction_data.py` | `exact_match_rate.png`, `extraction_counts.png` — exact-match rate and counts per feature (top 20). |
| `performance_patient/performance_patient.py` | `performance_patient.png` — F1/semantic score vs. training-set size. |
| `performance_representation/performance_representation.py` | `performance_vs_frequency.png` — per-feature score vs. feature frequency. |

## Setup

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the whole pipeline was tested with
(Python 3.11, CUDA 12.6, one 48 GB GPU; training Phi-3.5-mini peaks around
9 GB of VRAM). Training logs to Weights & Biases; set `WANDB_MODE=disabled`
to train without an account. For 4-bit GGUF inference, also install
`llama-cpp-python` built with CUDA support.

## Known issues

- **`performance_patient.py`** plots hand-transcribed constants rather than
  reading from a results file, so it will silently go stale.
- Several `data_scripts/` files build a `results`/analysis list intended
  for export that's populated but never written — looks like an unfinished
  feature.
- All scripts take their table number / prompting mode / model as variables
  edited by hand at the top of the file (or in the `__main__` block) rather
  than CLI arguments — easy to accidentally run against the wrong table.
- The figure scripts in `graph/` read fixed file names
  (`result_llama_no_prompt6_*.xlsx`, `extraction_features_no_prompt6_all.xlsx`);
  edit them to point at your own runs.

## License

The code in this repository is released under the [MIT License](LICENSE).
The models it fine-tunes keep their own licenses (Llama 3.1 Community
License, Apache 2.0 for Qwen 2.5, MIT for Phi-3.5), which apply to any model
trained with it.
