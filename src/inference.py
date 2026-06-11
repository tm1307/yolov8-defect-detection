# Expected runtime: ~0.1s per image (T4 GPU), ~1s per image (CPU)
# Tested on: Colab T4 / local CPU
"""
inference.py — Single-image and batch inference for defect detection.

This module provides production-ready inference utilities:
1. Single image prediction with annotated output
2. Batch inference over a directory of images
3. Results serialization to JSON for downstream processing
4. Visualization with bounding boxes, labels, and confidence scores

All functions use the Ultralytics YOLO predict API and return structured
results rather than raw tensors, making them easy to integrate with
web apps and REST APIs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """A single detected object with its bounding box and metadata.

    Attributes:
        class_id: Integer class index.
        class_name: Human-readable class name.
        confidence: Detection confidence score in [0, 1].
        bbox_xyxy: Bounding box as [x1, y1, x2, y2] in pixels.
        bbox_xywh: Bounding box as [x_center, y_center, width, height] in pixels.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_xywh: list[float]


@dataclass
class InferenceResult:
    """Complete inference result for a single image.

    Attributes:
        image_path: Path to the source image.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        num_detections: Total number of objects detected.
        detections: List of Detection objects.
        inference_time_ms: Time taken for inference in milliseconds.
    """

    image_path: str
    image_width: int
    image_height: int
    num_detections: int
    detections: list[Detection]
    inference_time_ms: float


# ---------------------------------------------------------------------------
# Color palette for visualization
# ---------------------------------------------------------------------------
# Distinct colors for up to 10 classes (BGR format for OpenCV)
CLASS_COLORS: list[tuple[int, int, int]] = [
    (255, 85, 85),    # Red
    (85, 255, 85),    # Green
    (85, 85, 255),    # Blue
    (255, 255, 85),   # Yellow
    (255, 85, 255),   # Magenta
    (85, 255, 255),   # Cyan
    (255, 170, 85),   # Orange
    (170, 85, 255),   # Purple
    (85, 255, 170),   # Mint
    (255, 85, 170),   # Pink
]

# Visualization constants
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LABEL_PADDING = 5


def load_config(config_path: str) -> dict[str, Any]:
    """Load inference configuration from a YAML file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str) -> YOLO:
    """Load a trained YOLOv8 model from a checkpoint file.

    Args:
        model_path: Path to the .pt weights file.

    Returns:
        Loaded YOLO model ready for inference.

    Raises:
        FileNotFoundError: If the weights file doesn't exist.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(model_path)
    print(f"[inference] Model loaded from: {model_path}")
    return model


def predict_single(
    model: YOLO,
    image_path: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
) -> InferenceResult:
    """Run inference on a single image.

    Args:
        model: Loaded YOLO model.
        image_path: Path to the input image.
        confidence_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for Non-Maximum Suppression (NMS).
        max_detections: Maximum number of detections to return.

    Returns:
        InferenceResult containing all detections and metadata.

    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the image cannot be read by OpenCV.
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_height, img_width = image.shape[:2]

    # Run inference
    start_time = time.perf_counter()
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        iou=iou_threshold,
        max_det=max_detections,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Parse results
    detections: list[Detection] = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        class_names = model.names  # {int: str} mapping from the model

        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            xywh = boxes.xywh[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())
            cls_name = class_names.get(cls_id, f"class_{cls_id}")

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox_xyxy=xyxy,
                    bbox_xywh=xywh,
                )
            )

    return InferenceResult(
        image_path=str(image_path),
        image_width=img_width,
        image_height=img_height,
        num_detections=len(detections),
        detections=detections,
        inference_time_ms=round(elapsed_ms, 2),
    )


def predict_batch(
    model: YOLO,
    image_dir: str,
    confidence_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> list[InferenceResult]:
    """Run inference on all images in a directory.

    Args:
        model: Loaded YOLO model.
        image_dir: Path to directory containing images.
        confidence_threshold: Minimum confidence threshold.
        iou_threshold: IoU threshold for NMS.
        max_detections: Maximum detections per image.
        extensions: Tuple of valid image file extensions.

    Returns:
        List of InferenceResult objects, one per image.

    Raises:
        FileNotFoundError: If the image directory doesn't exist.
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = sorted(
        f for f in image_dir_path.iterdir()
        if f.suffix.lower() in extensions
    )

    if len(image_files) == 0:
        print(f"[inference] No images found in {image_dir} with extensions {extensions}")
        return []

    print(f"[inference] Running batch inference on {len(image_files)} images...")

    results: list[InferenceResult] = []
    for img_path in image_files:
        result = predict_single(
            model=model,
            image_path=str(img_path),
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )
        results.append(result)

    # Summary statistics
    total_detections = sum(r.num_detections for r in results)
    avg_time = np.mean([r.inference_time_ms for r in results])
    print(f"[inference] Batch complete: {len(results)} images, "
          f"{total_detections} total detections, "
          f"{avg_time:.1f}ms avg inference time")

    return results


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: Input image as a NumPy array (BGR format).
        detections: List of Detection objects to visualize.

    Returns:
        Annotated image as a NumPy array (BGR format).
    """
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox_xyxy]
        color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        # Prepare label text
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]

        # Draw label background
        label_y1 = max(y1 - label_size[1] - 2 * LABEL_PADDING, 0)
        label_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + label_size[0] + 2 * LABEL_PADDING, label_y2),
            color,
            -1,  # Filled rectangle
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1 + LABEL_PADDING, y1 - LABEL_PADDING),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),  # White text
            FONT_THICKNESS,
        )

    return annotated


def save_annotated_image(
    image_path: str,
    detections: list[Detection],
    output_path: str,
) -> str:
    """Load an image, draw detections on it, and save the result.

    Args:
        image_path: Path to the original image.
        detections: List of Detection objects.
        output_path: Path to save the annotated image.

    Returns:
        Path to the saved annotated image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    annotated = draw_detections(image, detections)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)
    print(f"[inference] Annotated image saved to: {output_path}")

    return output_path


def results_to_json(
    results: list[InferenceResult],
    output_path: str,
) -> str:
    """Serialize inference results to a JSON file.

    Args:
        results: List of InferenceResult objects.
        output_path: Path to write the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    serializable = [asdict(r) for r in results]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[inference] Results saved to: {output_path}")

    return output_path


def run_inference(
    model_path: str,
    input_path: str,
    config_path: str = "configs/default.yaml",
    save_annotated: bool = True,
    save_json: bool = True,
) -> list[InferenceResult]:
    """End-to-end inference pipeline for single image or batch.

    Automatically detects whether the input is a file or directory and
    runs single or batch inference accordingly.

    Args:
        model_path: Path to trained model weights.
        input_path: Path to a single image or a directory of images.
        config_path: Path to configuration YAML.
        save_annotated: Whether to save annotated images.
        save_json: Whether to save results as JSON.

    Returns:
        List of InferenceResult objects.
    """
    config = load_config(config_path)
    inf_cfg = config["inference"]

    model = load_model(model_path)
    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
        # Single image
        result = predict_single(
            model=model,
            image_path=str(input_path_obj),
            confidence_threshold=inf_cfg["confidence_threshold"],
            iou_threshold=inf_cfg["iou_threshold"],
            max_detections=inf_cfg["max_detections"],
        )
        all_results = [result]
        print(f"[inference] {result.num_detections} detections in {result.inference_time_ms:.1f}ms")

    elif input_path_obj.is_dir():
        # Batch inference
        all_results = predict_batch(
            model=model,
            image_dir=str(input_path_obj),
            confidence_threshold=inf_cfg["confidence_threshold"],
            iou_threshold=inf_cfg["iou_threshold"],
            max_detections=inf_cfg["max_detections"],
        )

    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # --- Save outputs ---
    output_dir = Path(inf_cfg["output_dir"])

    if save_annotated:
        for result in all_results:
            img_name = Path(result.image_path).stem
            out_path = str(output_dir / "annotated" / f"{img_name}_annotated.jpg")
            save_annotated_image(result.image_path, result.detections, out_path)

    if save_json:
        json_path = str(output_dir / "results.json")
        results_to_json(all_results, json_path)

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run defect detection inference on images."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single image or directory of images.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML.",
    )
    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Skip saving annotated images.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Skip saving results JSON.",
    )
    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        input_path=args.input,
        config_path=args.config,
        save_annotated=not args.no_annotated,
        save_json=not args.no_json,
    )
