# Expected runtime: 2-5 min (validation on 500 images, Colab T4)
# Tested on: Colab T4 / local CPU
"""
evaluate.py — Model evaluation with mAP@50, F1 score, and confusion matrix.

This module provides:
1. Standard COCO-style mAP evaluation via Ultralytics' built-in validator
2. Per-class precision, recall, and F1 computation
3. Confusion matrix visualization and export
4. A formatted results summary table for the README

The evaluation uses Ultralytics' internal metrics engine, which implements
the same mAP calculation as the official COCO evaluation toolkit. We extract
and reformat those metrics for our use case.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict[str, Any]:
    """Load evaluation configuration from a YAML file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_validation(
    model_path: str,
    dataset_yaml: str,
    config: dict[str, Any],
) -> Any:
    """Run YOLOv8 validation and return the results object.

    This wraps model.val() with our configured thresholds. The returned
    object contains all metrics computed by Ultralytics.

    Args:
        model_path: Path to the trained model weights (.pt file).
        dataset_yaml: Path to the dataset.yaml file.
        config: Configuration dictionary with evaluation thresholds.

    Returns:
        Ultralytics validation Results object containing metrics.

    Raises:
        FileNotFoundError: If model weights or dataset YAML don't exist.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

    eval_cfg = config["evaluation"]

    model = YOLO(model_path)
    results = model.val(
        data=dataset_yaml,
        conf=eval_cfg["confidence_threshold"],
        iou=eval_cfg["iou_threshold"],
        max_det=eval_cfg["max_detections"],
        verbose=True,
        plots=True,  # Ultralytics will save PR curves, confusion matrix, etc.
    )

    return results


def extract_metrics(results: Any, class_names: list[str]) -> dict[str, Any]:
    """Extract and structure key metrics from Ultralytics results.

    Parses the results object to produce a clean dictionary with:
    - Overall mAP@50, mAP@50:95
    - Per-class precision, recall, F1
    - Detection counts

    Args:
        results: Ultralytics validation results object.
        class_names: Ordered list of class names.

    Returns:
        Structured dictionary of evaluation metrics.

    Note:
        F1 is computed as the harmonic mean of precision and recall.
        These are micro-averaged metrics from the best confidence threshold
        chosen by Ultralytics' internal optimization.
    """
    # Access the metrics box object
    # Ultralytics stores per-class metrics in results.box
    box = results.box

    metrics: dict[str, Any] = {
        "overall": {
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "mean_precision": float(box.mp),
            "mean_recall": float(box.mr),
            "mean_f1": _compute_f1(float(box.mp), float(box.mr)),
        },
        "per_class": {},
    }

    # Per-class metrics
    # box.ap50() returns per-class AP@50 as a numpy array
    ap50_per_class = box.ap50
    precision_per_class = box.p  # Per-class precision at best threshold
    recall_per_class = box.r  # Per-class recall at best threshold

    for idx, class_name in enumerate(class_names):
        if idx < len(ap50_per_class):
            p = float(precision_per_class[idx])
            r = float(recall_per_class[idx])
            metrics["per_class"][class_name] = {
                "AP50": float(ap50_per_class[idx]),
                "precision": p,
                "recall": r,
                "F1": _compute_f1(p, r),
            }

    return metrics


def _compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall.

    Args:
        precision: Precision value in [0, 1].
        recall: Recall value in [0, 1].

    Returns:
        F1 score in [0, 1]. Returns 0.0 if both inputs are zero.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def plot_confusion_matrix(
    results: Any,
    class_names: list[str],
    output_path: str,
    normalize: bool = True,
) -> str:
    """Generate and save a confusion matrix heatmap.

    Uses the confusion matrix computed during validation by Ultralytics,
    then renders it as a clean seaborn heatmap.

    Args:
        results: Ultralytics validation results containing confusion matrix.
        class_names: Ordered list of class names for axis labels.
        output_path: File path to save the figure (e.g., .png or .pdf).
        normalize: If True, normalize rows to show percentages.

    Returns:
        Path to the saved figure.

    Note:
        The confusion matrix from Ultralytics includes a 'background' class
        as the last row/column. We include it in the plot for completeness,
        as it shows false positive and false negative rates.
    """
    # Ultralytics stores the confusion matrix in results.confusion_matrix
    cm = results.confusion_matrix
    # The .matrix attribute gives the raw numpy array
    matrix = cm.matrix

    display_names = class_names + ["background"]

    if normalize:
        # Normalize by row (true class) to get percentages
        row_sums = matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else ".0f",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("Confusion Matrix — YOLOv8 Defect Detection", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved to: {output_path}")

    return output_path


def format_results_table(metrics: dict[str, Any]) -> str:
    """Format evaluation metrics as a Markdown table for the README.

    Args:
        metrics: Structured metrics dictionary from extract_metrics().

    Returns:
        A Markdown-formatted table string.
    """
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@50 | {metrics['overall']['mAP50']:.4f} |",
        f"| mAP@50:95 | {metrics['overall']['mAP50_95']:.4f} |",
        f"| Mean Precision | {metrics['overall']['mean_precision']:.4f} |",
        f"| Mean Recall | {metrics['overall']['mean_recall']:.4f} |",
        f"| Mean F1 | {metrics['overall']['mean_f1']:.4f} |",
        "",
        "### Per-Class Results",
        "",
        "| Class | AP@50 | Precision | Recall | F1 |",
        "|-------|-------|-----------|--------|-----|",
    ]

    for class_name, class_metrics in metrics["per_class"].items():
        lines.append(
            f"| {class_name} | {class_metrics['AP50']:.4f} | "
            f"{class_metrics['precision']:.4f} | {class_metrics['recall']:.4f} | "
            f"{class_metrics['F1']:.4f} |"
        )

    return "\n".join(lines)


def save_metrics_json(metrics: dict[str, Any], output_path: str) -> str:
    """Save evaluation metrics to a JSON file for programmatic access.

    Args:
        metrics: Structured metrics dictionary.
        output_path: Path to write the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved to: {output_path}")
    return output_path


def evaluate(
    model_path: str,
    dataset_yaml: str,
    config_path: str = "configs/default.yaml",
) -> dict[str, Any]:
    """Full evaluation pipeline: validate, extract metrics, plot, save.

    This is the main entry point for model evaluation. It:
    1. Runs YOLO validation on the val split
    2. Extracts mAP, precision, recall, F1 per class
    3. Generates and saves a confusion matrix
    4. Prints a formatted results table
    5. Saves metrics to JSON

    Args:
        model_path: Path to trained model weights (.pt file).
        dataset_yaml: Path to dataset.yaml for Ultralytics.
        config_path: Path to configuration YAML.

    Returns:
        Dictionary of all evaluation metrics.
    """
    config = load_config(config_path)
    class_names = config["data"]["subset_categories"]

    print(f"[evaluate] Model: {model_path}")
    print(f"[evaluate] Dataset: {dataset_yaml}")
    print(f"[evaluate] Classes: {class_names}")

    # --- Run validation ---
    print("\n[evaluate] Running validation...")
    results = run_validation(model_path, dataset_yaml, config)

    # --- Extract metrics ---
    metrics = extract_metrics(results, class_names)

    # --- Print results table ---
    table = format_results_table(metrics)
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(table)
    print(f"{'='*60}\n")

    # --- Save confusion matrix ---
    figures_dir = config["paths"]["figures_dir"]
    cm_path = plot_confusion_matrix(
        results,
        class_names,
        output_path=str(Path(figures_dir) / "confusion_matrix.png"),
    )

    # --- Save metrics JSON ---
    metrics_json_path = save_metrics_json(
        metrics,
        output_path=str(Path(figures_dir) / "evaluation_metrics.json"),
    )

    # --- Save markdown table ---
    table_path = Path(figures_dir) / "results_table.md"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"[evaluate] Results table saved to: {table_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLOv8 defect detection model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset.yaml.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration YAML.",
    )
    args = parser.parse_args()

    metrics = evaluate(
        model_path=args.model,
        dataset_yaml=args.dataset,
        config_path=args.config,
    )
