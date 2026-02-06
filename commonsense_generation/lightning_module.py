"""PyTorch Lightning module for LLaVA-Next fine-tuning."""

import re
from functools import partial

import lightning as L
import numpy as np
import torch
from nltk import edit_distance
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig, DataConfig
from .dataset import LlavaDataset, train_collate_fn, eval_collate_fn


class LlavaModelPLModule(L.LightningModule):
    """Lightning module wrapping LLaVA-Next for training and evaluation.

    Training uses teacher-forcing cross-entropy loss.
    Evaluation uses autoregressive generation with normalized edit distance.
    """

    def __init__(
        self,
        model,
        processor,
        train_dataset: LlavaDataset,
        val_dataset: LlavaDataset,
        model_cfg: ModelConfig,
        training_cfg: TrainingConfig,
        data_cfg: DataConfig,
    ):
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_cfg = model_cfg
        self.training_cfg = training_cfg
        self.data_cfg = data_cfg

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, pixel_values, image_sizes, labels = batch
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            labels=labels,
        )
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        input_ids, attention_mask, pixel_values, image_sizes, answers = batch

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            max_new_tokens=self.model_cfg.max_length,
        )
        predictions = self.processor.batch_decode(
            generated_ids[:, input_ids.size(1):], skip_special_tokens=True
        )

        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            ned = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(ned)

            if self.training_cfg.verbose and len(scores) == 1:
                print(f"Prediction: {pred}")
                print(f"    Answer: {answer}")
                print(f" Normed ED: {scores[0]}")

        self.log("val_edit_distance", np.mean(scores))
        return scores

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.training_cfg.lr)

    def train_dataloader(self):
        collate = partial(
            train_collate_fn,
            processor=self.processor,
            max_length=self.model_cfg.max_length,
        )
        return DataLoader(
            self.train_dataset,
            collate_fn=collate,
            batch_size=self.training_cfg.batch_size,
            shuffle=True,
            num_workers=self.data_cfg.num_workers,
        )

    def val_dataloader(self):
        collate = partial(
            eval_collate_fn,
            processor=self.processor,
            max_length=self.model_cfg.max_length,
        )
        return DataLoader(
            self.val_dataset,
            collate_fn=collate,
            batch_size=self.training_cfg.batch_size,
            shuffle=False,
            num_workers=self.data_cfg.num_workers,
        )
