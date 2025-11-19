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

### Training a Tokenizer

```bash
cd baseline
python train_tokenizer.py
```

This will:
1. Read text data from `../dataset/pretrain.jsonl`
2. Train a BPE tokenizer with vocab size 6400
3. Save the tokenizer to `../model/tokenizer.json`

### Pretraining the Model

```bash
cd baseline
python train_pretrain.py \
    --data_path ../dataset/pretrain.jsonl \
    --tokenizer_path ../dataset/tokenizer \
    --epochs 1 \
    --batch_size 32 \
    --learning_rate 5e-4
```

This will:
1. Load tokenizer from the specified path
2. Load training data from JSONL file using `MinimindDataset` from `dataset.py`
3. Train the MiniMind model using the current branch's model architecture

## Changes from Master Branch

The scripts have been adapted to work with the current branch structure:
- Uses `MinimindDataset` from `dataset.py` instead of `PretrainDataset` from master's `dataset.lm_dataset`
- Uses `MiniMindConfig` and `MiniMindForCausalLM` from `minimind.py` instead of master's `model.model_minimind`
- Includes simplified helper functions instead of importing from master's `trainer.trainer_utils`
- Data path changed from `pretrain_hq.jsonl` to `pretrain.jsonl` to match current branch conventions

## Requirements

- PyTorch
- transformers
- tokenizers
- langchain-text-splitters (for dataset preparation)
