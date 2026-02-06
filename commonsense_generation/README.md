# Commonsense Generation Module

This module fine-tunes [LLaVA-Next](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf) to generate structured commonsense annotations for memes. The fine-tuned model produces a meme description, identifies up to 5 commonsense violation categories, and recommends an intervention.

## Module Structure

```
commonsense_generation/
├── finetune_llava_next.py   # CLI entry point for fine-tuning
├── config.py                # Configuration dataclasses and prompt template
├── dataset.py               # LlavaDataset class and collate functions
├── model.py                 # Model loading with LoRA / QLoRA / full fine-tuning
├── lightning_module.py      # PyTorch Lightning training module
├── callbacks.py             # PushToHubCallback for HF Hub integration
├── inference_llava_next.py  # Inference script for generating commonsense annotations
└── data_processing.py       # Data preparation (train/val split, metadata creation)
```

## Prerequisites

Follow the installation steps in the [main README](../README.md), then ensure these additional dependencies are available:

```bash
pip install peft bitsandbytes lightning wandb nltk
```

## Data Preparation

The fine-tuning script expects a Hugging Face `imagefolder` dataset layout:

```
data_dir/
├── train/
│   ├── metadata.csv       # columns: file_name, text
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── validation/
    ├── metadata.csv
    ├── image_101.jpg
    └── ...
```

Each `metadata.csv` maps image filenames to ground-truth commonsense annotations. Use `data_processing.py` to create this layout from raw CSV files:

```bash
python data_processing.py
```

## Fine-Tuning

### Quick start (QLoRA, default settings)

```bash
python -m commonsense_generation.finetune_llava_next \
    --data_dir path/to/your/data/
```

### Full example with all options

```bash
python -m commonsense_generation.finetune_llava_next \
    --data_dir path/to/your/data/ \
    --model_id llava-hf/llava-v1.6-mistral-7b-hf \
    --use_qlora \
    --lora_r 4 \
    --lora_alpha 8 \
    --max_length 256 \
    --max_epochs 10 \
    --lr 1e-4 \
    --batch_size 1 \
    --accumulate_grad_batches 8 \
    --devices 0 \
    --precision 16-mixed \
    --save_path saved_model_detailed_prompt \
    --use_wandb \
    --wandb_project LLaVaNeXT \
    --wandb_name my-run \
    --early_stopping \
    --early_stopping_patience 3
```

### CLI Arguments

#### Model
| Argument | Default | Description |
|----------|---------|-------------|
| `--model_id` | `llava-hf/llava-v1.6-mistral-7b-hf` | HF model ID for LLaVA-Next |
| `--use_lora` | `False` | Use LoRA (float16 base + adapters) |
| `--use_qlora` | `True` | Use QLoRA (4-bit base + adapters) |
| `--lora_r` | `4` | LoRA rank |
| `--lora_alpha` | `8` | LoRA alpha scaling |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--max_length` | `256` | Max token sequence length |

#### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--max_epochs` | `10` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate (AdamW) |
| `--batch_size` | `1` | Per-device batch size |
| `--accumulate_grad_batches` | `8` | Gradient accumulation steps |
| `--gradient_clip_val` | `1.0` | Gradient clipping threshold |
| `--precision` | `16-mixed` | Training precision |
| `--devices` | `0` | GPU device(s) to use |
| `--limit_val_batches` | `2` | Number of validation batches per epoch |
| `--verbose` | `True` | Print sample predictions during validation |

#### Data
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `../commonsense_labelled_data_new/` | Path to imagefolder dataset |
| `--num_workers` | `4` | DataLoader worker count |

#### Logging & Saving
| Argument | Default | Description |
|----------|---------|-------------|
| `--save_path` | `saved_model_detailed_prompt` | Directory to save the fine-tuned model |
| `--use_wandb` | `False` | Enable Weights & Biases logging |
| `--wandb_project` | `LLaVaNeXT` | W&B project name |
| `--wandb_name` | `llava-next-demo-cord` | W&B run name |
| `--push_to_hub` | `False` | Push model to HF Hub after each epoch |
| `--repo_id` | `YOUR-HUB-REPO-TO-PUSH` | HF Hub repository ID |
| `--early_stopping` | `False` | Enable early stopping on val edit distance |
| `--early_stopping_patience` | `3` | Epochs to wait before stopping |

## Inference

After fine-tuning, use the inference script to generate commonsense annotations for new images:

```bash
python inference_llava_next.py
```

Update the `SAVE_PATH` and image paths in `inference_llava_next.py` to point to your fine-tuned model and target images.

## Training Modes

The script supports three training configurations:

| Mode | Memory | Speed | Quality |
|------|--------|-------|---------|
| **QLoRA** (`--use_qlora`) | Lowest (~6 GB) | Moderate | Good |
| **LoRA** (`--use_lora`) | Medium (~14 GB) | Moderate | Good |
| **Full fine-tuning** (neither flag) | Highest (~126 GB) | Fastest | Best |

QLoRA is the default and recommended mode for consumer GPUs.
