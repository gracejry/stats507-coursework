from typing import Dict, List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import evaluate as hf_evaluate

from config import ADAPTER_PATH, DEVICE
from data import load_oxford_pets
from model import generate_caption, load_lora_for_inference
from transformers import BlipForConditionalGeneration
from config import MODEL_ID

def fine_grained_accuracy(model, dataset, id2label: Dict[int, str], sample_size: int = 200) -> float:
    print("=== Starting Quantitative Evaluation ===")
    correct = 0
    print(f"Evaluating on {sample_size} random samples from validation set...")

    for i in tqdm(range(sample_size)):
        idx = int(len(dataset) * 0.9) + i
        item = dataset[idx]
        true_label = id2label[item["label"]]
        pred_caption = generate_caption(model, processor=None, image=item["image"])  # processor is inside helper

        if true_label.lower() in pred_caption.lower():
            correct += 1

    acc = correct / sample_size
    print(f"Fine-Grained Breed Accuracy: {acc * 100:.2f}%")
    return acc

def zero_shot_baseline(processor, dataset, id2label: Dict[int, str], sample_size: int = 200) -> float:
    print("=== Running Zero-Shot Baseline (Original Model) ===")
    base_model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    base_model.eval()

    correct = 0
    print(f"Evaluating {sample_size} samples on Base Model...")

    for i in tqdm(range(sample_size)):
        idx = int(len(dataset) * 0.9) + i
        item = dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        true_label = id2label[item["label"]]

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        out = base_model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)

        if true_label.lower() in caption.lower():
            correct += 1

    acc = correct / sample_size
    print(f"Zero-Shot Baseline Accuracy: {acc * 100:.2f}%")

    del base_model
    torch.cuda.empty_cache()

    return acc

def confusion_analysis(model, processor, raw_dataset, id2label: Dict[int, str]) -> None:
    model.eval()
    y_true: List[str] = []
    y_pred: List[str] = []

    print("Generating predictions for Confusion Matrix (this takes a moment)...")
    for i in tqdm(range(200)):
        idx = int(len(raw_dataset) * 0.9) + i
        item = raw_dataset[idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        true_name = id2label[item["label"]]
        y_true.append(true_name)

        inputs = processor(images=image, return_tensors="pt").to(DEVICE)
        out = model.generate(**inputs, max_length=50)
        caption = processor.decode(out[0], skip_special_tokens=True)

        found_breed = "Unknown"
        for name in id2label.values():
            if name.lower() in caption.lower():
                found_breed = name
                break
        y_pred.append(found_breed)

    breed_names = list(id2label.values()) + ["Unknown"]
    cm = confusion_matrix(y_true, y_pred, labels=breed_names)

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, xticklabels=breed_names, yticklabels=breed_names, cmap="Blues", annot=False)
    plt.title("Confusion Matrix: Fine-Grained Pet Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Top confused pairs
    np.fill_diagonal(cm, 0)
    pairs = []
    for i in range(len(breed_names)):
        for j in range(len(breed_names)):
            if cm[i, j] > 0:
                pairs.append((breed_names[i], breed_names[j], cm[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n=== Top Confused Pairs (Analysis Insight) ===")
    print("Format: True Breed -> Misclassified as -> Count")
    for t, p, c in pairs[:10]:
        print(f"{t} -> {p} : {c} times")

def caption_metrics_and_failure(model, processor, eval_dataset, id2label):
    print("Loading metrics...")
    bleu = hf_evaluate.load("bleu")
    rouge = hf_evaluate.load("rouge")

    references, predictions, true_breeds = [], [], []

    model.eval()
    target_data = getattr(eval_dataset, "ds", eval_dataset)

    print("Generating captions for full evaluation set (Calculating BLEU/ROUGE)...")
    for item in tqdm(target_data):
        image = item["image"]
        label_idx = int(item["label"])
        breed_name = id2label[label_idx]

        ref_text = f"a photo of a {breed_name}, a type of pet."

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(DEVICE)
            out = model.generate(**inputs, max_length=50)
            pred_text = processor.decode(out[0], skip_special_tokens=True)

        references.append([ref_text])
        predictions.append(pred_text)
        true_breeds.append(breed_name)

    print("\nComputing scores...")
    bleu_results = bleu.compute(predictions=predictions, references=references)
    rouge_results = rouge.compute(predictions=predictions, references=references)

    print("\n========================================")
    print("RESULTS REPORT (Proposal Requirement)")
    print("========================================")
    print(f"BLEU Score: {bleu_results['bleu']:.4f}")
    print(f"ROUGE-L Score: {rouge_results['rougeL']:.4f}")
    print("========================================")

    # Failure analysis
    failures = [
        (true, pred)
        for true, pred in zip(true_breeds, predictions)
        if true.lower() not in pred.lower()
    ]
    print("\n=== Failure Analysis (Qualitative) ===")
    print(f"Total Failures: {len(failures)} out of {len(predictions)}")
    print("\nTop 5 Failure Examples (Where the model got it wrong):")
    for i, (true, pred) in enumerate(failures[:5], start=1):
        print(f"Example {i}:")
        print(f" - True Breed: [{true}]")
        print(f" - Generated: [{pred}]")
        print("-" * 30)

    # Data distribution plots
    plt.figure(figsize=(12, 6))
    breed_counts = pd.Series(true_breeds).value_counts().head(20)
    sns.barplot(x=breed_counts.index, y=breed_counts.values)
    plt.xticks(rotation=90)
    plt.title("Top 20 Breed Distribution in Validation Set")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    pred_lengths = [len(p.split()) for p in predictions]
    plt.figure(figsize=(8, 5))
    sns.histplot(pred_lengths, bins=15, kde=True)
    plt.title("Distribution of Generated Caption Lengths (Word Count)")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.show()

def main() -> None:
    raw_dataset, id2label, processor, train_dataset, eval_dataset = load_oxford_pets()
    model = load_lora_for_inference(ADAPTER_PATH)

    fine_grained_accuracy(model, raw_dataset, id2label, sample_size=200)

    zs_acc = zero_shot_baseline(processor, raw_dataset, id2label, sample_size=200)
    print(f"Zero-shot baseline acc: {zs_acc * 100:.2f}%")

    confusion_analysis(model, processor, raw_dataset, id2label)
    caption_metrics_and_failure(model, processor, eval_dataset, id2label)


if __name__ == "__main__":
    main()
