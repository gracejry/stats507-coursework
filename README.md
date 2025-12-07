# stats507-coursework
Data Science and Analytics using Python

# BLIP LoRA Fine-Tuning on Oxford-IIIT Pet Dataset

This repository contains a refactored version of a Colab notebook that
fine-tunes **Salesforce/blip-image-captioning-base** on the
**Oxford-IIIT Pet** dataset using **LoRA** adapters.

The code is organised into multiple Python modules so that it can be
easily used as a GitHub project (training, evaluation, and analysis
scripts).

## 1. Environment

```bash
git clone https://github.com/<your-name>/blip-oxford-pets-lora.git
cd blip-oxford-pets-lora

pip install -r requirements.txt
