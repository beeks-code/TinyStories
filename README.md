# TinyStories GPT

A GPT-2 style language model built from scratch and trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset.

## About
This project implements a transformer-based language model from scratch using PyTorch, trained on the TinyStories dataset to generate coherent short stories.

## Project Structure
TinyStories/

├── model/          # GPT architecture (attention, transformer blocks)

├── data/           # Dataset and dataloader

├── training/       # Training loop and loss functions

├── inference/      # Text generation

├── configs/        # Model configuration

├── utils/          # Metrics and plotting

├── train.py        # Main training script

├── retrain.py      # Continue training from checkpoint

└── test.py         # Run inference

## Model
- Architecture: GPT-2 style transformer
- Parameters: ~162M
- Context length: 256 tokens
- Training data: 20k TinyStories

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Train:
```bash
python train.py
```
Test:
```bash
python test.py
```

## Built By
Bikram — built and trained from scratch as a learning project.
