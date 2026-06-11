# Expected runtime: N/A (module file)
# Tested on: Colab T4 / local CPU
"""
Source package for YOLOv8 Defect Detection pipeline.

Modules:
    data_loader  — COCO subset downloading, filtering, and augmentation
    train        — YOLOv8 fine-tuning with W&B experiment tracking
    evaluate     — mAP, F1, and confusion matrix computation
    inference    — Single-image and batch inference utilities
    app          — Gradio web demo for interactive defect detection
"""
