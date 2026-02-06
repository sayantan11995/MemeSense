"""Dataset and collate functions for LLaVA-Next fine-tuning."""

from typing import Dict, List, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset

from .config import PROMPT_TEMPLATE


class LlavaDataset(Dataset):
    """PyTorch Dataset for LLaVA-Next fine-tuning.

    Loads an image-folder dataset where each sample has an image and
    corresponding ground-truth text annotation.
    """

    def __init__(self, data_dir: str, split: str = "train"):
        super().__init__()
        self.dataset = load_dataset("imagefolder", data_dir=data_dir, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        sample = self.dataset[idx]
        return sample["image"], sample["text"]


def train_collate_fn(examples, processor, max_length: int):
    """Collate function for training batches.

    Prepares (image, ground_truth) pairs into model inputs with labels.
    """
    images = []
    texts = []
    for image, ground_truth in examples:
        images.append(image)
        prompt = f"[INST] <image>\n{PROMPT_TEMPLATE} [\\INST]"
        texts.append(prompt)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["image_sizes"],
        batch["labels"],
    )


def eval_collate_fn(examples, processor, max_length: int):
    """Collate function for evaluation batches.

    Only encodes the prompt (no ground truth) for autoregressive generation.
    """
    images = []
    texts = []
    answers = []
    for image, ground_truth in examples:
        images.append(image)
        prompt = f"[INST] <image>\n{PROMPT_TEMPLATE} [\\INST]"
        texts.append(prompt)
        answers.append(ground_truth)

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    return (
        batch["input_ids"],
        batch["attention_mask"],
        batch["pixel_values"],
        batch["image_sizes"],
        answers,
    )
