import torch

MODEL_ID: str = "Salesforce/blip-image-captioning-base"

SEED: int = 42

OUTPUT_DIR: str = "./blip_oxford_lora_checkpoints"
ADAPTER_PATH: str = "./blip-oxford-pets-lora-adapter"

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16: bool = torch.cuda.is_bf16_supported()

print(f"Running on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Device Architecture: {DEVICE}")
print(f"BF16 Precision Supported: {USE_BF16}")

def set_seed(seed: int = SEED) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
