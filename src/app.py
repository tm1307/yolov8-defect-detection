
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

from src.inference import (
    CLASS_COLORS,
    Detection,
    draw_detections,
    predict_single,
)

MODEL: YOLO | None = None
MODEL_PATH: str = ""

def load_model_for_app(model_path: str) -> YOLO:
    global MODEL, MODEL_PATH

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}\n"
            f"Train a model first with: python -m src.train"
        )

    MODEL = YOLO(model_path)
    MODEL_PATH = model_path
    print(f"[app] Model loaded: {model_path}")
    print(f"[app] Classes: {MODEL.names}")
    return MODEL

def predict_and_annotate(
    image: np.ndarray | None,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray | None, str]:
    if image is None:
        return None, "⚠️ Please upload an image."

    if MODEL is None:
        return None, "❌ Model not loaded. Check server logs."

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = MODEL.predict(
        source=image_bgr,
        conf=confidence_threshold,
        iou=iou_threshold,
        max_det=100,
        verbose=False,
    )

    detections: list[Detection] = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            xywh = boxes.xywh[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = MODEL.names.get(cls_id, f"class_{cls_id}")

            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox_xyxy=xyxy,
                bbox_xywh=xywh,
            ))

    annotated_bgr = draw_detections(image_bgr, detections)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    if len(detections) == 0:
        summary = "No defects detected in this image."
    else:
        summary_lines = [
            f"
            "|
            "|---|-------|------------|---------------------|",
        ]
        for idx, det in enumerate(detections, 1):
            bbox_str = ", ".join(f"{c:.0f}" for c in det.bbox_xyxy)
            summary_lines.append(
                f"| {idx} | {det.class_name} | {det.confidence:.3f} | ({bbox_str}) |"
            )

        class_counts: dict[str, int] = {}
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1

        summary_lines.append(f"\n**Class Distribution:** " +
                           ", ".join(f"{k}: {v}" for k, v in sorted(class_counts.items())))

        summary = "\n".join(summary_lines)

    return annotated_rgb, summary

def build_gradio_app() -> gr.Blocks:
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }

            Upload a product image to detect defects. Adjust the sliders to
            control detection sensitivity.
            ---
            **Model:** YOLOv8n (nano) · **Dataset:** COCO 2017 subset
            · **Classes:** bottle, cup, bowl, knife, scissors

            *See [GitHub repo](https://github.com/tm1307/yolov8-defect-detection)
            for training details and full pipeline.*
