

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
    """Load the YOLO model for the Gradio app.

    This is called once at startup. The model is stored in a module-level
    variable and reused for all predictions.

    Args:
        model_path: Path to trained model weights.

    Returns:
        Loaded YOLO model.
    """
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
    """Run detection on an uploaded image and return annotated result.

    This is the main Gradio callback function. It:
    1. Validates the input image
    2. Runs YOLO inference with user-specified thresholds
    3. Draws bounding boxes on the image
    4. Formats a detection summary table

    Args:
        image: Input image as a NumPy array (RGB format from Gradio).
        confidence_threshold: Minimum confidence for detections.
        iou_threshold: IoU threshold for NMS.

    Returns:
        Tuple of (annotated_image, detection_summary_text).
    """
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
            f"### 🔍 {len(detections)} Detection(s) Found\n",
            "| # | Class | Confidence | BBox (x1,y1,x2,y2) |",
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
    """Construct the Gradio Blocks UI layout.

    Returns:
        A Gradio Blocks application ready to launch.
    """
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    """

    with gr.Blocks(
        title="YOLOv8 Defect Detection",
        css=custom_css,
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """

            Upload a product image to detect defects. Adjust the sliders to
            control detection sensitivity.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="numpy",
                    sources=["upload", "webcam"],
                    height=400,
                )

                with gr.Row():
                    conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Higher = fewer but more confident detections",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.45,
                        step=0.05,
                        label="IoU Threshold (NMS)",
                        info="Lower = more aggressive duplicate suppression",
                    )

                detect_button = gr.Button(
                    "🔍 Detect Defects",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Detection Results",
                    type="numpy",
                    height=400,
                )
                detection_summary = gr.Markdown(
                    label="Detection Summary",
                    value="Upload an image and click 'Detect Defects' to begin.",
                )

        detect_button.click(
            fn=predict_and_annotate,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, detection_summary],
        )

        input_image.change(
            fn=predict_and_annotate,
            inputs=[input_image, conf_slider, iou_slider],
            outputs=[output_image, detection_summary],
        )

        gr.Markdown(
            """
            ---
            **Model:** YOLOv8n (nano) · **Dataset:** COCO 2017 subset
            · **Classes:** bottle, cup, bowl, knife, scissors

            *See [GitHub repo](https://github.com/tm1307/yolov8-defect-detection)
            for training details and full pipeline.*
            """
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch Gradio demo for defect detection."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/train_run/weights/best.pt",
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (useful for Colab).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on.",
    )
    args = parser.parse_args()

    load_model_for_app(args.model)

    app = build_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
