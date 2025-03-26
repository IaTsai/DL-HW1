# NYCU Computer Vision 2025 Spring HW1

**StudentID:** 313553058
**Name:** IanTsai

---

## Introduction

This project explores the performance of various architectures based on **ResNet50V2** with advanced attention mechanisms and transfer learning techniques.

### Architecture Overview

The best-performing configuration in our experiments is:
**ResNet50V2 + Frz(stem+stage1) + GCBlock(stage3+4) + SE(stage3) + Dropout(0.3)**

- **GCBlock** (Global Context Block) is applied at Stages 3 and 4 to enhance long-range context aggregation.
- **SEBlock** (Squeeze-and-Excitation) is applied at Stage 3 for channel-wise feature recalibration.
- **Dropout** (‘p=0.3’) is added after normalization to reduce overfitting.
- **stem** and **stage1** are *frozen* to preserve pre-trained low-level features from ImageNet.

---

## How to Install

You can recreate the environment using the provided file:

```bash
conda create --name _ResNet_env --file environment.txt
conda activate _ResNet_env
```

---

## How to Run

1. Run the main training and inference script using the following command:

```bash
CUDA_VISIBLE_DEVICES=2 python DL-HW1.py
```

This will train the model and generate the initial prediction file named `prediction.csv`.
2. Before submission, you need to correct label encoding issues caused by `ImageFolder`.
First, **rename** the generated file:

```bash
mv prediction.csv org-prediction.csv
```

Then, run the conversion script to fix the label mismatch:

```bash
python Convert.py
```

This will generate a corrected `prediction.csv` file suitable for submission.
