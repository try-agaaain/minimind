# Baseline Scripts

This directory contains baseline training scripts adapted from the master branch to work with the current branch structure.

## Files

- `train_tokenizer.py` - Tokenizer training script (adapted from master branch `scripts/train_tokenizer.py`)
- `train_pretrain.py` - Pretraining script (adapted from master branch `trainer/train_pretrain.py`)

## Data Loading

Both scripts are configured to load data from the `dataset` directory:
- **train_tokenizer.py**: Loads from `../dataset/pretrain.jsonl` (JSONL format with 'text' field)
- **train_pretrain.py**: Loads from `../dataset/pretrain.jsonl` (JSONL format with 'token_ids' field)

## Usage

### Training a Tokenizer (Baseline Mode)

```bash
cd baseline
python train_tokenizer.py
```

This will:
1. Read text data from `../dataset/pretrain.jsonl`
2. Train a BPE tokenizer with vocab size 6400
3. Save the tokenizer to `../model/tokenizer.json`

### Pretraining the Model

#### Option 1: Standard Mode (Automatic Dataset Preparation - **New!**)

The new **standard mode** automatically checks for tokenizer and dataset files. If they don't exist, it will prepare them automatically using the `NovelDatasetPreparator` (similar to `train.py`).

```bash
cd baseline
python train_pretrain.py \
    --tokenizer_mode standard \
    --dataset_dir ../dataset \
    --data_path ../dataset/pretrain.jsonl \
    --tokenizer_path ../dataset/tokenizer \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4
```

This will:
1. Check if tokenizer and dataset exist
2. If not, automatically prepare them from `.txt` files in `dataset_dir`
3. Load tokenizer and dataset
4. Train the MiniMind model

**New Parameters:**
- `--tokenizer_mode {baseline,standard}` - Select tokenizer preparation method (default: `standard`)
- `--dataset_dir DIR` - Directory containing `.txt` files for dataset preparation (default: `../dataset`)
- `--chunk_size SIZE` - Text chunk size for splitting (default: `1024`)
- `--chunk_overlap SIZE` - Overlap between chunks (default: `128`)
- `--vocab_size SIZE` - Vocabulary size for tokenizer (default: `6400`)

#### Option 2: Baseline Mode (Manual Preparation)

The **baseline mode** requires you to manually prepare tokenizer and dataset files before training (original behavior).

```bash
cd baseline
python train_pretrain.py \
    --tokenizer_mode baseline \
    --data_path ../dataset/pretrain.jsonl \
    --tokenizer_path ../dataset/tokenizer \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4
```

This will:
1. Check if tokenizer and dataset exist (raises error if missing)
2. Load tokenizer and dataset
3. Train the MiniMind model

**Note:** In baseline mode, you must run `train_tokenizer.py` first to create the tokenizer.

## Changes from Master Branch

The scripts have been adapted to work with the current branch structure:
- Uses `MinimindDataset` from `dataset.py` instead of `PretrainDataset` from master's `dataset.lm_dataset`
- Uses `MiniMindConfig` and `MiniMindForCausalLM` from `minimind.py` instead of master's `model.model_minimind`
- Includes simplified helper functions instead of importing from master's `trainer.trainer_utils`
- Data path changed from `pretrain_hq.jsonl` to `pretrain.jsonl` to match current branch conventions
- **NEW:** Added automatic dataset preparation mode similar to `train.py`
- **NEW:** Added `--tokenizer_mode` parameter to choose between baseline and standard tokenizer methods

## Requirements

- PyTorch
- transformers
- tokenizers
- langchain-text-splitters (for dataset preparation)
