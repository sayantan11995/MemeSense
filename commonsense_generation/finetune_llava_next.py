"""Fine-tune LLaVA-Next for meme commonsense annotation generation.

This script fine-tunes the LLaVA-Next model using QLoRA (or LoRA / full
fine-tuning) on an image-folder dataset of memes paired with commonsense
annotations.  It uses PyTorch Lightning for the training loop and supports
W&B logging and Hugging Face Hub model pushing.

Usage:
    python -m commonsense_generation.finetune_llava_next \
        --data_dir ../commonsense_labelled_data_new/ \
        --max_epochs 10 \
        --batch_size 1 \
        --save_path saved_model_detailed_prompt
"""

import argparse

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from .callbacks import PushToHubCallback
from .config import DataConfig, HubConfig, ModelConfig, TrainingConfig
from .dataset import LlavaDataset
from .lightning_module import LlavaModelPLModule
from .model import apply_peft, load_model, load_processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune LLaVA-Next for meme commonsense generation"
    )

    # Model
    parser.add_argument("--model_id", type=str, default=ModelConfig.model_id)
    parser.add_argument("--use_lora", action="store_true", default=ModelConfig.use_lora)
    parser.add_argument("--use_qlora", action="store_true", default=ModelConfig.use_qlora)
    parser.add_argument("--lora_r", type=int, default=ModelConfig.lora_r)
    parser.add_argument("--lora_alpha", type=int, default=ModelConfig.lora_alpha)
    parser.add_argument("--lora_dropout", type=float, default=ModelConfig.lora_dropout)
    parser.add_argument("--max_length", type=int, default=ModelConfig.max_length)

    # Training
    parser.add_argument("--max_epochs", type=int, default=TrainingConfig.max_epochs)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--batch_size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--accumulate_grad_batches", type=int, default=TrainingConfig.accumulate_grad_batches)
    parser.add_argument("--gradient_clip_val", type=float, default=TrainingConfig.gradient_clip_val)
    parser.add_argument("--precision", type=str, default=TrainingConfig.precision)
    parser.add_argument("--devices", type=int, nargs="+", default=TrainingConfig.devices)
    parser.add_argument("--limit_val_batches", type=int, default=TrainingConfig.limit_val_batches)
    parser.add_argument("--verbose", action="store_true", default=TrainingConfig.verbose)

    # Data
    parser.add_argument("--data_dir", type=str, default=DataConfig.data_dir)
    parser.add_argument("--num_workers", type=int, default=DataConfig.num_workers)

    # Hub / logging
    parser.add_argument("--save_path", type=str, default=HubConfig.save_path)
    parser.add_argument("--repo_id", type=str, default=HubConfig.repo_id)
    parser.add_argument("--wandb_project", type=str, default=HubConfig.wandb_project)
    parser.add_argument("--wandb_name", type=str, default=HubConfig.wandb_name)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)

    return parser.parse_args()


def main():
    args = parse_args()

    # Build config objects from CLI args
    model_cfg = ModelConfig(
        model_id=args.model_id,
        use_lora=args.use_lora,
        use_qlora=args.use_qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
    )
    training_cfg = TrainingConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        devices=args.devices,
        limit_val_batches=args.limit_val_batches,
        verbose=args.verbose,
    )
    data_cfg = DataConfig(data_dir=args.data_dir, num_workers=args.num_workers)
    hub_cfg = HubConfig(
        repo_id=args.repo_id,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        save_path=args.save_path,
    )

    # Load processor and model
    processor = load_processor(model_cfg)
    model = load_model(model_cfg)

    if model_cfg.use_lora or model_cfg.use_qlora:
        model = apply_peft(model, model_cfg)

    # Load datasets
    train_dataset = LlavaDataset(data_cfg.data_dir, split="train")
    val_dataset = LlavaDataset(data_cfg.data_dir, split="validation")

    # Build Lightning module
    model_module = LlavaModelPLModule(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        data_cfg=data_cfg,
    )

    # Callbacks
    callbacks = []
    if args.push_to_hub:
        callbacks.append(PushToHubCallback(hub_cfg.repo_id))
    if args.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_edit_distance",
                patience=args.early_stopping_patience,
                verbose=False,
                mode="min",
            )
        )

    # Logger
    logger = None
    if args.use_wandb:
        logger = WandbLogger(project=hub_cfg.wandb_project, name=hub_cfg.wandb_name)

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=training_cfg.devices,
        max_epochs=training_cfg.max_epochs,
        accumulate_grad_batches=training_cfg.accumulate_grad_batches,
        check_val_every_n_epoch=training_cfg.check_val_every_n_epoch,
        gradient_clip_val=training_cfg.gradient_clip_val,
        precision=training_cfg.precision,
        limit_val_batches=training_cfg.limit_val_batches,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks if callbacks else None,
    )

    # Train
    trainer.fit(model_module)

    # Save
    model_module.model.save_pretrained(hub_cfg.save_path)
    print(f"Model saved to {hub_cfg.save_path}")


if __name__ == "__main__":
    main()
