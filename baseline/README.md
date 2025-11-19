# Baseline Scripts

This directory contains baseline training scripts from the master branch of the minimind repository.

## Files

- `train_tokenizer.py` - Tokenizer training script from `scripts/train_tokenizer.py` in master branch
- `train_pretrain.py` - Pretraining script from `trainer/train_pretrain.py` in master branch

## Data Loading

Both scripts are configured to load data from the `dataset` directory:
- Data path: `../dataset/pretrain_hq.jsonl` (relative to baseline directory)

## Note

These scripts serve as baseline references. They may require additional dependencies and module structure from the master branch to run directly. The scripts are included here as a reference point for fine-tuning work.
