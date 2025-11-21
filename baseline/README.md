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

Both **standard** and **baseline** modes now automatically prepare missing tokenizer and dataset files. The difference is in the tokenizer training approach:

#### Option 1: Standard Mode (Uses NovelDatasetPreparator Tokenizer - Default)

The **standard mode** uses the `NovelDatasetPreparator` approach (similar to `train.py`) for both dataset and tokenizer preparation.

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
2. If not, automatically prepare them using `NovelDatasetPreparator` (BPE tokenizer with standard special tokens)
3. Load tokenizer and dataset
4. Train the MiniMind model

**New Parameters:**
- `--tokenizer_mode {baseline,standard}` - Select tokenizer preparation method (default: `standard`)
- `--dataset_dir DIR` - Directory containing `.txt` files for dataset preparation (default: `../dataset`)
- `--chunk_size SIZE` - Text chunk size for splitting (default: `1024`)
- `--chunk_overlap SIZE` - Overlap between chunks (default: `128`)
- `--vocab_size SIZE` - Vocabulary size for tokenizer (default: `6400`)

#### Option 2: Baseline Mode (Uses Baseline Tokenizer Training)

The **baseline mode** uses the baseline tokenizer training approach from `train_tokenizer.py` (BPE with special tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`).

```bash
cd baseline
python train_pretrain.py \
    --tokenizer_mode baseline \
    --dataset_dir ../dataset \
    --data_path ../dataset/pretrain.jsonl \
    --tokenizer_path ../dataset/tokenizer \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4
```

This will:
1. Check if dataset exists
2. If not, automatically prepare dataset using `NovelDatasetPreparator`
3. Check if tokenizer exists
4. If not, automatically train tokenizer using baseline approach (`train_tokenizer.py` method)
5. Load tokenizer and dataset
6. Train the MiniMind model

**Note:** Both modes now automatically prepare missing files. The key difference is the tokenizer training method used.

## Changes from Master Branch

The scripts have been adapted to work with the current branch structure:
- Uses `MinimindDataset` from `dataset.py` instead of `PretrainDataset` from master's `dataset.lm_dataset`
- Uses `MiniMindConfig` and `MiniMindForCausalLM` from `minimind.py` instead of master's `model.model_minimind`
- Includes simplified helper functions instead of importing from master's `trainer.trainer_utils`
- Data path changed from `pretrain_hq.jsonl` to `pretrain.jsonl` to match current branch conventions
- **NEW:** Added automatic dataset preparation in both modes
- **NEW:** Added `--tokenizer_mode` parameter to choose between baseline and standard tokenizer methods
- **NEW:** Refactored `train_tokenizer.py` to be callable as a function from `train_pretrain.py`

## Requirements

- PyTorch
- transformers
- tokenizers
- langchain-text-splitters (for dataset preparation)
