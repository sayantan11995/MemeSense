"""Model loading and LoRA/QLoRA setup for LLaVA-Next."""

from typing import List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)

from .config import ModelConfig


def find_all_linear_names(model) -> List[str]:
    """Find all linear layer names suitable for LoRA adaptation.

    Excludes the vision encoder, multimodal projector, and lm_head
    so that LoRA adapters are applied only to the language model.
    """
    multimodal_keywords = ["multi_modal_projector", "vision_model"]
    lora_module_names = set()

    for name, module in model.named_modules():
        if any(kw in name for kw in multimodal_keywords):
            continue
        if isinstance(module, torch.nn.Linear):
            leaf_name = name.split(".")[-1] if "." in name else name
            lora_module_names.add(leaf_name)

    lora_module_names.discard("lm_head")
    return list(lora_module_names)


def load_processor(model_cfg: ModelConfig) -> AutoProcessor:
    """Load and configure the processor/tokenizer."""
    processor = AutoProcessor.from_pretrained(model_cfg.model_id)
    processor.tokenizer.padding_side = "right"
    return processor


def load_model(model_cfg: ModelConfig) -> LlavaNextForConditionalGeneration:
    """Load the LLaVA-Next model with optional LoRA/QLoRA.

    Supports three modes:
      - QLoRA: 4-bit quantized base model + LoRA adapters
      - LoRA: float16 base model + LoRA adapters
      - Full fine-tuning: float16 base model (all parameters trainable)
    """
    if model_cfg.use_qlora or model_cfg.use_lora:
        bnb_config = None
        if model_cfg.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_cfg.model_id,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_cfg.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    return model


def apply_peft(model, model_cfg: ModelConfig):
    """Apply LoRA adapters to the model and prepare for k-bit training."""
    lora_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model
