from typing import Dict, List
import torch
from transformers import Trainer, TrainingArguments
from config import ADAPTER_PATH, OUTPUT_DIR, USE_BF16, set_seed
from data import load_oxford_pets
from model import create_lora_model

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "attention_mask": attention_mask,
        "labels": input_ids,
    }

def plot_loss_curves(trainer: Trainer) -> None:
    import matplotlib.pyplot as plt

    log_history = trainer.state.log_history

    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for log in log_history:
        if "loss" in log:
            train_steps.append(log["step"])
            train_loss.append(log["loss"])
        if "eval_loss" in log:
            eval_steps.append(log["step"])
            eval_loss.append(log["eval_loss"])

    plt.figure(figsize=(10, 5))
    plt.plot(train_steps, train_loss, label="Training Loss")
    if eval_loss:
        plt.plot(eval_steps, eval_loss, label="Validation Loss", linestyle="--")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def main() -> None:
    set_seed()

    dataset, id2label, processor, train_dataset, eval_dataset = load_oxford_pets()

    model = create_lora_model()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=USE_BF16,
        fp16=False if USE_BF16 else True,
        dataloader_num_workers=4,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    print("Starting training loop...")
    trainer.train()

    model.save_pretrained(ADAPTER_PATH)
    print(f"Model adapter saved to {ADAPTER_PATH}")

    print("Plotting Loss Curves...")
    plot_loss_curves(trainer)


if __name__ == "__main__":
    main()
