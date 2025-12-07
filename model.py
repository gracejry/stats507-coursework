from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import BlipForConditionalGeneration
from transformers.models.blip.modeling_blip import (
    BlipForConditionalGeneration as BlipOriginal,
    BlipVisionModel,
)

from config import DEVICE, MODEL_ID


# ---------- Patch BLIP forward ----------

def _patch_blip_if_needed() -> None:
    """Patch BLIP vision & text forward to ignore extra kwargs."""
    if not hasattr(BlipVisionModel, "_patched"):
        _orig_vision_forward = BlipVisionModel.forward

        def safe_vision_forward(self, pixel_values, interpolate_pos_encoding=False, **kwargs):
            kwargs.pop("inputs_embeds", None)
            kwargs.pop("num_items_in_batch", None)
            return _orig_vision_forward(
                self,
                pixel_values,
                interpolate_pos_encoding=interpolate_pos_encoding,
                **kwargs,
            )

        BlipVisionModel.forward = safe_vision_forward
        BlipVisionModel._patched = True
        print("Applied patch for BlipVisionModel")

    if not hasattr(BlipOriginal, "_patched"):
        _orig_blip_forward = BlipOriginal.forward

        def safe_blip_forward(self, *args: Any, **kwargs: Any):
            kwargs.pop("num_items_in_batch", None)
            return _orig_blip_forward(self, *args, **kwargs)

        BlipOriginal.forward = safe_blip_forward
        BlipOriginal._patched = True
        print("Applied patch for BlipForConditionalGeneration")


# ---------- LoRA model helpers ----------

def create_lora_model() -> BlipForConditionalGeneration:
    """Create BLIP model with LoRA adapters, ready for training."""
    _patch_blip_if_needed()

    print("Loading Base Model...")
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["query", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)
    model.to(DEVICE)

    print("\n=== Trainable Parameters Analysis ===")
    model.print_trainable_parameters()
    return model


def load_lora_for_inference(adapter_path: str) -> PeftModel:
    """
    Load base BLIP + LoRA adapter for evaluation / inference.
    """
    _patch_blip_if_needed()
    base_model = BlipForConditionalGeneration.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.to(DEVICE)
    model.eval()
    return model


# ---------- Caption generation ----------

def generate_caption(model, processor, image, max_length: int = 50, num_beams: int = 5) -> str:
    """Generate caption for a single PIL image."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values,
            max_length=max_length,
            num_beams=num_beams,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
