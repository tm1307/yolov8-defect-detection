
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

@dataclass
class Detection:

    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    bbox_xywh: list[float]

@dataclass
class InferenceResult:

    image_path: str
    image_width: int
    image_height: int
    num_detections: int
    detections: list[Detection]
    inference_time_ms: float

CLASS_COLORS: list[tuple[int, int, int]] = [
    (255, 85, 85),
    (85, 255, 85),
    (85, 85, 255),
    (255, 255, 85),
    (255, 85, 255),
    (85, 255, 255),
    (255, 170, 85),
    (170, 85, 255),
    (85, 255, 170),
    (255, 85, 170),
]

BBOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
LABEL_PADDING = 5

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_path: str) -> YOLO:
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
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img_height, img_width = image.shape[:2]

    start_time = time.perf_counter()
    results = model.predict(
        source=image_path,
        conf=confidence_threshold,
        iou=iou_threshold,
        max_det=max_detections,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    detections: list[Detection] = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        class_names = model.names

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
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox_xyxy]
        color = CLASS_COLORS[det.class_id % len(CLASS_COLORS)]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        label = f"{det.class_name} {det.confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]

        label_y1 = max(y1 - label_size[1] - 2 * LABEL_PADDING, 0)
        label_y2 = y1
        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + label_size[0] + 2 * LABEL_PADDING, label_y2),
            color,
            -1,
        )

        cv2.putText(
            annotated,
            label,
            (x1 + LABEL_PADDING, y1 - LABEL_PADDING),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )

    return annotated

def save_annotated_image(
    image_path: str,
    detections: list[Detection],
    output_path: str,
) -> str:
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
    config = load_config(config_path)
    inf_cfg = config["inference"]

    model = load_model(model_path)
    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
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
        all_results = predict_batch(
            model=model,
            image_dir=str(input_path_obj),
            confidence_threshold=inf_cfg["confidence_threshold"],
            iou_threshold=inf_cfg["iou_threshold"],
            max_detections=inf_cfg["max_detections"],
        )

    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

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
