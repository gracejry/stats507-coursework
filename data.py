from typing import Dict, Tuple

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor

from config import MODEL_ID


class PetCaptionDataset(Dataset):
    """Wrap Oxford-IIIT Pet dataset for BLIP captioning."""

    def __init__(
        self,
        dataset,
        processor,
        id2label: Dict[int, str],
        split_ratio: float = 0.9,
        split: str = "train",
    ):
        self.processor = processor
        self.id2label = id2label

        train_size = int(len(dataset) * split_ratio)
        if split == "train":
            self.ds = dataset.select(range(train_size))
        else:
            self.ds = dataset.select(range(train_size, len(dataset)))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        breed_name = self.id2label[item["label"]]
        text = f"A photo of a {breed_name}, a type of pet."

        encoding = self.processor(
            images=image,
            text=text,
            padding="max_length",
            return_tensors="pt",
        )

        # squeeze batch dimension
        return {k: v.squeeze() for k, v in encoding.items()}


def load_oxford_pets(split_ratio: float = 0.9) -> Tuple:
    """
    Load Oxford-IIIT Pet dataset and return:
    (raw_dataset, id2label, processor, train_dataset, eval_dataset)
    """
    print("Downloading and loading Oxford-IIIT Pet Dataset (mirror)...")
    try:
        dataset = load_dataset("timm/oxford-iiit-pet", split="train").shuffle(seed=42)
    except Exception:
        print("timm mirror failed, trying pcuenq mirror...")
        dataset = load_dataset("pcuenq/oxford-pets", split="train").shuffle(seed=42)

    # label names
    if hasattr(dataset.features["label"], "names"):
        labels = dataset.features["label"].names
    else:
        labels = [f"Breed_{i}" for i in range(37)]
        print("Warning: Label names metadata missing, using generic placeholders.")

    id2label = {i: label.replace("_", " ").title() for i, label in enumerate(labels)}

    print(f"Total Images: {len(dataset)}")
    print(f"Sample Label Mapping: 0 -> {id2label[0]}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    train_dataset = PetCaptionDataset(dataset, processor, id2label, split_ratio, "train")
    eval_dataset = PetCaptionDataset(dataset, processor, id2label, split_ratio, "eval")

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Eval Set Size: {len(eval_dataset)}")
    print("Data loaded successfully from mirror.")

    return dataset, id2label, processor, train_dataset, eval_dataset
