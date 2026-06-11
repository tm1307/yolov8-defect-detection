# 🔬 Real-Time Product Defect Detection with YOLOv8

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3-red.svg)](https://pytorch.org/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLOv8-8.2-green.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **End-to-end ML pipeline for product defect detection**, featuring COCO-pretrained YOLOv8 fine-tuning, Albumentations augmentation, W&B experiment tracking, and a Gradio web demo — all runnable on a free Google Colab T4 GPU.

---

## 📋 Project Overview

This project implements a **real-time object defect detection system** using YOLOv8-nano fine-tuned on a curated subset of COCO 2017. The pipeline demonstrates production ML engineering practices:

- **Transfer learning** from COCO-pretrained weights to a domain-specific 5-class detector
- **Data engineering** with automated COCO→YOLO format conversion and Albumentations augmentation
- **Experiment tracking** with Weights & Biases (hyperparameters, metrics, model artifacts)
- **Evaluation** with mAP@50, per-class F1, and confusion matrix visualization
- **Serving** via an interactive Gradio web interface with adjustable detection thresholds

> **Note on dataset choice:** We use 5 COCO object categories (bottle, cup, bowl, knife, scissors) as **proxies** for product defect types. In a production system, this would be replaced with a domain-specific defect dataset (e.g., MVTec AD). The COCO proxy approach lets us demonstrate the full pipeline on freely available, well-annotated data without requiring proprietary defect images.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
│                                                             │
│  COCO 2017 ──► Filter by ──► COCO→YOLO ──► Albumentations  │
│  (train/val)   category      format         augmentation    │
│                (5 classes)   conversion     (flip, rotate,  │
│                                             blur, noise)    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   TRAINING PIPELINE                         │
│                                                             │
│  YOLOv8n ──► Fine-tune ──► W&B Logging ──► Checkpoint      │
│  (COCO       (50 epochs,    (metrics,       (best.pt,       │
│  pretrained)  AMP, SGD)     artifacts)      last.pt)        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  EVALUATION PIPELINE                        │
│                                                             │
│  best.pt ──► Validate ──► mAP@50 ──► Confusion  ──► JSON   │
│              (val set)     F1          Matrix         export │
│                            P/R         (seaborn)            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   SERVING PIPELINE                          │
│                                                             │
│  Gradio ──► Image Upload ──► YOLOv8 ──► Annotated ──► JSON  │
│  Web UI     or Webcam        Predict    Image        table   │
│  (:7860)                     (~100ms)   + BBoxes             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
yolov8-defect-detection/
├── configs/
│   └── default.yaml          # All hyperparameters and paths (no magic numbers)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # COCO subset → YOLO format + augmentation
│   ├── train.py              # YOLOv8 fine-tuning with W&B tracking
│   ├── evaluate.py           # mAP, F1, confusion matrix computation
│   ├── inference.py          # Single-image and batch inference
│   └── app.py                # Gradio web demo
├── notebooks/
│   └── colab_quickstart.ipynb  # One-click Colab notebook (TODO)
├── tests/
│   └── (unit tests)
├── assets/
│   └── demo.gif              # Demo GIF placeholder
├── datasets/                 # Created by data_loader.py (gitignored)
├── outputs/                  # Training outputs (gitignored)
├── requirements.txt          # Pinned dependencies
├── .gitignore
└── README.md                 # You are here
```

---

## 🚀 Setup Instructions

### Option 1: Google Colab (Recommended)

1. Open a new Colab notebook with **T4 GPU** runtime
2. Run the following cells:

```python
# Cell 1: Clone and install
!git clone https://github.com/tm1307/yolov8-defect-detection.git
%cd yolov8-defect-detection
!pip install -r requirements.txt

# Cell 2: Download COCO subset
!mkdir -p datasets/coco
!wget -q http://images.cocodataset.org/zips/train2017.zip -O datasets/coco/train2017.zip
!wget -q http://images.cocodataset.org/zips/val2017.zip -O datasets/coco/val2017.zip
!wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O datasets/coco/annotations.zip
!cd datasets/coco && unzip -q train2017.zip && unzip -q val2017.zip && unzip -q annotations.zip

# Cell 3: Prepare dataset
!python -m src.data_loader --config configs/default.yaml

# Cell 4: Train (takes ~30-60 min on T4)
!python -m src.train --config configs/default.yaml --dataset datasets/yolo_defect/dataset.yaml

# Cell 5: Evaluate
!python -m src.evaluate --model outputs/train_run/weights/best.pt --dataset datasets/yolo_defect/dataset.yaml

# Cell 6: Launch demo
!python -m src.app --model outputs/train_run/weights/best.pt --share
```

### Option 2: Local Setup

```bash
# Prerequisites: Python 3.10+, CUDA 11.8+ (optional, for GPU)

git clone https://github.com/tm1307/yolov8-defect-detection.git
cd yolov8-defect-detection

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🏋️ Training

### Prepare the dataset

```bash
python -m src.data_loader --config configs/default.yaml
```

This will:
- Load COCO 2017 train/val annotations
- Filter to 5 target categories (bottle, cup, bowl, knife, scissors)
- Convert bounding boxes from COCO format to YOLO format
- Apply Albumentations augmentations (2x augmented copies per training image)
- Generate `datasets/yolo_defect/dataset.yaml`

### Run training

```bash
# With W&B logging (set your API key first)
export WANDB_API_KEY="your-key-here"
python -m src.train --config configs/default.yaml --dataset datasets/yolo_defect/dataset.yaml

# Without W&B (offline mode)
python -m src.train --config configs/default.yaml --dataset datasets/yolo_defect/dataset.yaml
```

### Key training parameters (in `configs/default.yaml`)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | YOLOv8n (nano) | Fits in T4 16GB VRAM |
| Epochs | 50 | Sufficient for convergence with early stopping |
| Batch size | 16 | Max stable batch for T4 with AMP |
| Image size | 640×640 | Standard YOLO resolution |
| Optimizer | SGD | Default for YOLO, well-tuned momentum=0.937 |
| AMP | Enabled | ~2x speedup on T4 |
| Early stopping | patience=10 | Prevents overfitting |

---

## 📊 Evaluation Results

### Run evaluation

```bash
python -m src.evaluate \
    --model outputs/train_run/weights/best.pt \
    --dataset datasets/yolo_defect/dataset.yaml \
    --config configs/default.yaml
```

### Actual Evaluation Results (YOLOv8n, 11 epochs, ~2000 train images)

> **Note:** Training converged and stopped early at 11 epochs due to early stopping patience (no improvement).

| Metric | Value |
|--------|-------|
| mAP@50 | 0.3157 |
| mAP@50:95 | 0.2584 |
| Mean Precision | 0.5319 |
| Mean Recall | 0.1000 |
| Mean F1 | 0.1684 |

### Per-Class Results

| Class | AP@50 | Precision | Recall | F1 |
|-------|-------|-----------|--------|-----|
| bottle | 0.4388 | 0.6667 | 0.1930 | 0.2994 |
| cup | 0.4312 | 0.7595 | 0.1190 | 0.2058 |
| bowl | 0.5407 | 0.9000 | 0.1827 | 0.3038 |
| knife | 0.1676 | 0.3333 | 0.0054 | 0.0106 |
| scissors | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Per-Class Notes
- **bottle, cup, bowl** — These objects are well-represented in the COCO subset, achieving reasonable precision.
- **knife, scissors** — These classes performed poorly (0.0 mAP for scissors) due to very low instance counts (e.g., only 19 scissors instances in the validation set) and their high intra-class variation. More targeted training data would be needed for these specific defects.

### Outputs

Evaluation produces:
- `outputs/figures/confusion_matrix.png` — Normalized confusion matrix heatmap
- `outputs/figures/evaluation_metrics.json` — Machine-readable metrics
- `outputs/figures/results_table.md` — Markdown table for documentation

---

## 🔍 Inference

### Single image

```bash
python -m src.inference \
    --model outputs/train_run/weights/best.pt \
    --input path/to/image.jpg
```

### Batch inference (directory)

```bash
python -m src.inference \
    --model outputs/train_run/weights/best.pt \
    --input path/to/image_directory/
```

### Output

- Annotated images saved to `outputs/inference/annotated/`
- JSON results saved to `outputs/inference/results.json`

---

## 🖥️ Web Demo

```bash
# Local
python -m src.app --model outputs/train_run/weights/best.pt

# Colab (creates a public share link)
python -m src.app --model outputs/train_run/weights/best.pt --share
```

Opens a Gradio interface at `http://localhost:7860` with:
- 📸 Image upload or webcam capture
- 🎛️ Adjustable confidence and IoU threshold sliders
- 📊 Detection results table with class, confidence, and bounding box coordinates

### Demo

<!-- Replace with your actual demo GIF -->
![Demo GIF](assets/demo.gif)

*Record your demo GIF by screen-recording the Gradio interface, then replace the placeholder above.*

---

## 🛠️ MLOps Features

| Feature | Tool | Purpose |
|---------|------|---------|
| Experiment tracking | W&B | Log metrics, hyperparams, model artifacts |
| Config management | YAML | Single source of truth for all parameters |
| Data versioning | W&B Artifacts | Track dataset versions (extensible) |
| Model checkpointing | Ultralytics | Auto-save best/last weights |
| Augmentation | Albumentations | Reproducible, bbox-safe transforms |
| Evaluation | Custom + Ultralytics | mAP, F1, confusion matrix |
| Serving | Gradio | Interactive web demo |
| Reproducibility | Seed + deterministic mode | Consistent results across runs |

---

## 📝 Limitations & Future Work

### Current Limitations
- **Proxy dataset:** COCO categories are not real manufacturing defects. For production use, replace with domain-specific data (MVTec AD, custom labeled data).
- **Model size:** YOLOv8n is optimized for speed over accuracy. Larger variants (s, m, l) would improve mAP but require more VRAM.
- **Training data:** ~2000 images is a small dataset. Performance would improve significantly with 10K+ labeled images.

### Future Extensions
- [ ] Replace COCO proxy with MVTec AD or custom defect dataset
- [ ] Add ONNX/TensorRT export for production deployment
- [ ] Implement FastAPI REST endpoint alongside Gradio
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Integrate DVC for data versioning
- [ ] Add model explainability (Grad-CAM heatmaps)

---

## 📚 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [COCO Dataset](https://cocodataset.org/)
- [Albumentations](https://albumentations.ai/)
- [Weights & Biases](https://wandb.ai/)
- Jocher, G., et al. "YOLOv8 by Ultralytics." (2023). GitHub: ultralytics/ultralytics.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


