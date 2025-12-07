# stats507-coursework
Data Science and Analytics using Python

# BLIP + LoRA on Oxford-IIIT Pet  
Fine-tuning an image captioning model for fine-grained pet breeds

> This repository contains the code and experiments for my STATS 507 final project,  
> where I fine-tune a BLIP image captioning model with LoRA on the Oxford-IIIT Pet dataset.

## 1. Project Overview

The goal of this project is to **adapt a general-purpose image captioning model** to the **fine-grained pet classification** setting:

- **Base model**: `Salesforce/blip-image-captioning-base`
- **Dataset**: Oxford-IIIT Pet (37 cat & dog breeds)
- **Technique**: Parameter-efficient fine-tuning using **LoRA** adapters
- **Task**: Generate captions that correctly mention the specific pet breed in the image

### Research Questions

1. Can LoRA fine-tuning help BLIP reliably generate captions that include the **correct breed name**?
2. How does the **fine-tuned model** compare to the **original (zero-shot) BLIP**?
3. Where does the model fail, and what patterns appear in **confusion matrix** and **attention maps**?

## 2. Repository Structure

```text
.
├── README.md                
├── requirements.txt         # Python dependencies
├── config.py                # Global configuration (paths, device, model ID, seed)
├── data.py                  # Dataset loading + custom Dataset wrapper
├── model.py                 # BLIP + LoRA model, patching, caption generation helper
├── train.py                 # Training script for LoRA fine-tuning
├── evaluate.py              # Quantitative evaluation & analysis
└── explainability.py        # Attention-based visualizations for interpretability
```


### File Descriptions

- **`config.py`**
  - Global configuration
  - Model ID, device, output directories
  - Mixed precision flags
  - Random seed settings

- **`data.py`**
  - Loads the Oxford-IIIT Pet dataset
  - Generates `id2label` mapping
  - Implements `PetCaptionDataset`
  - Builds prompts:
    > A photo of a {breed_name}, a type of pet.

- **`model.py`**
  - Patches BLIP for stable forward pass
  - Builds LoRA-enabled BLIP with:
    - rank = 32  
    - lora_alpha = 64  
    - dropout = 0.05  
    - target modules: `"query"`, `"value"`
  - Provides:
    - `create_lora_model()`
    - `load_lora_for_inference()`
    - `generate_caption()`

- **`train.py`**
  - Loads dataset and model
  - Configures HuggingFace `TrainingArguments`
  - Runs LoRA fine-tuning
  - Tracks loss
  - Saves trained LoRA adapter
  - Plots training & evaluation curves

- **`evaluate.py`**
  - Computes fine-grained breed accuracy
  - Runs zero-shot BLIP baseline
  - Produces confusion matrix
  - Identifies top confused breed pairs
  - Computes BLEU / ROUGE-L
  - Performs failure analysis
  - Generates distribution plots

- **`explainability.py`**
  - Extracts BLIP cross-attention
  - Creates patch-level attention heatmaps
  - Overlays heatmaps on images
  - Visualizes model focus during prediction

## 3. Installation
- Clone the repository:
  ```bash
  git clone https://github.com/<your-username>/<repo>.git
  cd <repo>
  ```

- Install Dependencies：
  ```bash
  pip install -r requirements.txt
  ```
  This will install:
  * `torch`, `transformers`, `datasets`, `peft`, `accelerate`
  * `evaluate`, `scikit-learn`, `matplotlib`, `seaborn`, `opencv-python`, `tqdm`
    
  Note: A GPU (CUDA) is strongly recommended. Training on CPU will be very slow.

## 4. Data & Model

No manual download is required.
The scripts automatically fetch everything from Hugging Face:

* Model: Salesforce/blip-image-captioning-base

* Dataset: Oxford-IIIT Pet (via Hugging Face Datasets mirror)

Dataset loading and preprocessing are handled inside `data.py`, while the base model is loaded in `model.py`.

## 5. How to Use

- Train the LoRA adapter:
  ```bash
  python train.py
  ```
  
This will:
- load and preprocess the dataset
- initialize BLIP + LoRA
- run multi-epoch training
- save the LoRA adapter to the directory defined in config.py

- Evaluate the fine-tuned model:
  ```bash
  python evaluate.py

The evaluation pipeline includes:
- fine-grained breed accuracy
- zero-shot BLIP baseline
- confusion matrix + top confused breed pairs
- BLEU / ROUGE-L scores
- failure case examples
- breed distribution & caption length plots

- Visualize attention heatmaps:
  ```bash
  python explainability.py

This script will:
- compute BLIP cross-attention
- generate attention heatmaps
- overlay heatmaps on the original images
- provide visual interpretability for model predictions

## 6. Configuration

- All global settings are defined in config.py:
  - `MODEL_ID`
  - output directories (OUTPUT_DIR, ADAPTER_PATH)
  - device configuration (cuda, cpu)
  - bf16 / fp16 precision flags
  - random seed
- Adjust hardware-related settings when needed:
  - lower batch size if GPU memory is limited
  - disable bf16 if unsupported
  - reduce number of epochs for faster experiments

## 7. Reproducing Project Results

Run LoRA training:
```bash
python train.py
```
Run full evaluation:
```bash
python evaluate.py
```
Generate attention maps:
```bash
python explainability.py
```
These three steps reproduce the metrics, confusion matrix, caption quality scores, and interpretability results documented in the project analysis.

## 8. Acknowledgements

- Model:
  - `Salesforce/blip-image-captioning-base`
- Dataset:
  - Oxford-IIIT Pet (via Hugging Face Datasets)
- LoRA library:
  - HuggingFace PEFT
- Training & evaluation framework:
  - HuggingFace Transformers

