
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
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_validation(
    model_path: str,
    dataset_yaml: str,
    config: dict[str, Any],
) -> Any:
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
        plots=True,
    )

    return results

def extract_metrics(results: Any, class_names: list[str]) -> dict[str, Any]:
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

    ap50_per_class = box.ap50
    precision_per_class = box.p
    recall_per_class = box.r

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
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def plot_confusion_matrix(
    results: Any,
    class_names: list[str],
    output_path: str,
    normalize: bool = True,
) -> str:
    cm = results.confusion_matrix
    matrix = cm.matrix

    display_names = class_names + ["background"]

    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
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
    lines = [
        "| Metric | Value |",
        "|--------|-------|",
        f"| mAP@50 | {metrics['overall']['mAP50']:.4f} |",
        f"| mAP@50:95 | {metrics['overall']['mAP50_95']:.4f} |",
        f"| Mean Precision | {metrics['overall']['mean_precision']:.4f} |",
        f"| Mean Recall | {metrics['overall']['mean_recall']:.4f} |",
        f"| Mean F1 | {metrics['overall']['mean_f1']:.4f} |",
        "",
        "
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
    config = load_config(config_path)
    class_names = config["data"]["subset_categories"]

    print(f"[evaluate] Model: {model_path}")
    print(f"[evaluate] Dataset: {dataset_yaml}")
    print(f"[evaluate] Classes: {class_names}")

    print("\n[evaluate] Running validation...")
    results = run_validation(model_path, dataset_yaml, config)

    metrics = extract_metrics(results, class_names)

    table = format_results_table(metrics)
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(table)
    print(f"{'='*60}\n")

    figures_dir = config["paths"]["figures_dir"]
    cm_path = plot_confusion_matrix(
        results,
        class_names,
        output_path=str(Path(figures_dir) / "confusion_matrix.png"),
    )

    metrics_json_path = save_metrics_json(
        metrics,
        output_path=str(Path(figures_dir) / "evaluation_metrics.json"),
    )

    table_path = Path(figures_dir) / "results_table.md"
    with open(table_path, "w") as f:
        f.write(table)
    print(f"[evaluate] Results table saved to: {table_path}")

    return metrics

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
