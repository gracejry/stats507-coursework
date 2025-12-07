import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

from config import ADAPTER_PATH, DEVICE
from data import load_oxford_pets
from model import load_lora_for_inference


def visualize_attention(model, processor, image, breed_name: str):
    """Return image overlaid with attention heatmap."""
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    text_prompt = f"A photo of a {breed_name}"
    text_inputs = processor.tokenizer(text_prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(
            pixel_values=inputs.pixel_values,
            input_ids=text_inputs.input_ids,
            output_attentions=True,
        )

    if not hasattr(outputs, "cross_attentions"):
        print("Warning: Cross attentions not found in outputs.")
        return np.array(image)

    last_layer_attn = outputs.cross_attentions[-1]
    avg_attn = last_layer_attn.mean(dim=1)  # [B, T, P]
    breed_attn = avg_attn[0, -2:, :].mean(dim=0)

    num_patches = breed_attn.shape[0]
    grid_size = int(np.sqrt(num_patches))
    attn_map = breed_attn.reshape(grid_size, grid_size).cpu().numpy()

    img_np = np.array(image)
    attn_map = cv2.resize(attn_map, (img_np.shape[1], img_np.shape[0]))
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255.0

    cam = heatmap + np.float32(img_np) / 255.0
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def main() -> None:
    raw_dataset, id2label, processor, train_dataset, eval_dataset = load_oxford_pets()
    model = load_lora_for_inference(ADAPTER_PATH)

    print("Generating Attention Maps for 3 Random Samples...")
    indices = np.random.choice(len(eval_dataset), 3, replace=False)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        raw_idx = int(len(raw_dataset) * 0.9) + int(idx)
        item = raw_dataset[raw_idx]
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        breed = id2label[item["label"]]

        try:
            viz_img = visualize_attention(model, processor, image, breed)
            plt.subplot(1, 3, i + 1)
            plt.imshow(viz_img)
            plt.title(f"Focus for '{breed}'", fontsize=12, fontweight="bold")
            plt.axis("off")
        except Exception as e:
            print(f"Skipped image due to error: {e}")
            plt.subplot(1, 3, i + 1)
            plt.imshow(image)
            plt.title(f"{breed} (Viz Error)", fontsize=10)
            plt.axis("off")

    plt.tight_layout()
    plt.show()
    print("Interpretation: Red/Yellow areas indicate high attention weights.")


if __name__ == "__main__":
    main()
