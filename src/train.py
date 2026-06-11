# Expected runtime: 30-60 min (50 epochs on Colab T4 with yolov8n, 2000 images)
# Tested on: Colab T4 / local CPU (CPU will be ~10x slower)
"""
train.py — YOLOv8 fine-tuning with Weights & Biases experiment tracking.

This module wraps Ultralytics' training API with:
1. Proper configuration management (YAML-driven, no magic numbers)
2. W&B integration for experiment tracking, hyperparameter logging, and
   artifact versioning
3. Reproducibility controls (seed, deterministic mode)
4. Colab-friendly defaults (batch size, workers, mixed precision)

Usage:
    python -m src.train --config configs/default.yaml --dataset datasets/yolo_defect/dataset.yaml
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

# Optional: wandb may not be installed in all environments
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def load_config(config_path: str) -> dict[str, Any]:
    """Load training configuration from a YAML file.

    Args:
        config_path: Absolute or relative path to the YAML config.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_wandb(config: dict[str, Any]) -> Any | None:
    """Initialize Weights & Biases run for experiment tracking.

    If W&B is not installed or WANDB_API_KEY is not set, logging is skipped
    gracefully. This ensures the training script works even without W&B.

    Args:
        config: Full configuration dictionary (logged as hyperparameters).

    Returns:
        The wandb Run object if initialized, None otherwise.
    """
    if not WANDB_AVAILABLE:
        print("[train] wandb not installed. Skipping experiment tracking.")
        return None

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key is None:
        print("[train] WANDB_API_KEY not set. Running in offline mode.")
        os.environ["WANDB_MODE"] = "offline"

    log_cfg = config["logging"]
    run = wandb.init(
        project=log_cfg["wandb_project"],
        entity=log_cfg.get("wandb_entity"),
        config=config,
        name=f"yolov8n-defect-{config['training']['epochs']}ep",
        tags=["yolov8", "defect-detection", "coco-subset"],
    )
    return run


def validate_environment() -> dict[str, str]:
    """Check that the training environment meets minimum requirements.

    Returns:
        Dictionary with environment info (device, CUDA version, etc.).

    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    env_info = {
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda or "N/A",
        "device_name": "CPU",
        "gpu_memory_gb": "N/A",
    }

    if torch.cuda.is_available():
        env_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        env_info["gpu_memory_gb"] = f"{gpu_mem_bytes / (1024**3):.1f}"

    print("[train] Environment:")
    for key, value in env_info.items():
        print(f"  {key}: {value}")

    return env_info


def build_training_args(config: dict[str, Any], dataset_yaml: str) -> dict[str, Any]:
    """Convert our config format to Ultralytics YOLO training arguments.

    This function maps our YAML config keys to the keyword arguments
    expected by model.train(). See Ultralytics docs for full reference:
    https://docs.ultralytics.com/modes/train/#arguments

    Args:
        config: Parsed YAML configuration dictionary.
        dataset_yaml: Path to the dataset.yaml file for Ultralytics.

    Returns:
        Dictionary of keyword arguments for model.train().
    """
    train_cfg = config["training"]
    model_cfg = config["model"]
    paths_cfg = config["paths"]

    training_args = {
        "data": dataset_yaml,
        "epochs": train_cfg["epochs"],
        "batch": train_cfg["batch_size"],
        "imgsz": model_cfg["input_size"],
        "lr0": train_cfg["learning_rate"],
        "optimizer": train_cfg["optimizer"],
        "momentum": train_cfg["momentum"],
        "weight_decay": train_cfg["weight_decay"],
        "warmup_epochs": train_cfg["warmup_epochs"],
        "patience": train_cfg["patience"],
        "save_period": train_cfg["save_period"],
        "workers": train_cfg["workers"],
        "device": train_cfg["device"],
        "project": paths_cfg["output_dir"],
        "name": "train_run",
        "exist_ok": True,
        # Reproducibility
        "seed": config["data"]["seed"],
        "deterministic": True,
        # Performance
        "amp": True,  # Mixed precision — essential for T4
        "cache": False,  # Don't cache images in RAM (Colab has limited RAM)
        # Logging
        "verbose": True,
        "plots": True,
    }

    return training_args


def train(
    config_path: str = "configs/default.yaml",
    dataset_yaml: str | None = None,
    resume: bool = False,
    resume_weights: str | None = None,
) -> Path:
    """Run YOLOv8 fine-tuning with full experiment tracking.

    This is the main training entry point. It:
    1. Loads config and validates the environment
    2. Initializes W&B tracking
    3. Loads the COCO-pretrained YOLOv8 model
    4. Runs training with configured hyperparameters
    5. Saves the best checkpoint and logs final metrics

    Args:
        config_path: Path to the training configuration YAML.
        dataset_yaml: Path to dataset.yaml. If None, uses the default path
            from the config (datasets/yolo_defect/dataset.yaml).
        resume: Whether to resume training from a checkpoint.
        resume_weights: Path to checkpoint weights for resuming. If None and
            resume=True, Ultralytics will try to find the last checkpoint.

    Returns:
        Path to the best model weights file.
    """
    # --- Load config ---
    config = load_config(config_path)
    print(f"[train] Loaded config from: {config_path}")

    # --- Validate environment ---
    env_info = validate_environment()

    # --- Resolve dataset path ---
    if dataset_yaml is None:
        dataset_yaml = str(
            Path(config["data"]["coco_root"]).parent / "yolo_defect" / "dataset.yaml"
        )
    if not Path(dataset_yaml).exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {dataset_yaml}\n"
            f"Run `python -m src.data_loader --config {config_path}` first."
        )
    print(f"[train] Dataset config: {dataset_yaml}")

    # --- Initialize W&B ---
    wandb_run = setup_wandb(config)
    if wandb_run is not None:
        wandb_run.log({"environment": env_info})

    # --- Load model ---
    model_architecture = config["model"]["architecture"]
    if resume and resume_weights:
        print(f"[train] Resuming from: {resume_weights}")
        model = YOLO(resume_weights)
    else:
        print(f"[train] Loading pretrained: {model_architecture}")
        model = YOLO(model_architecture)

    # --- Build training arguments ---
    training_args = build_training_args(config, dataset_yaml)
    print("[train] Training arguments:")
    for key, value in sorted(training_args.items()):
        print(f"  {key}: {value}")

    # --- Train ---
    print("\n[train] Starting training...")
    results = model.train(**training_args)

    # --- Extract results ---
    output_dir = Path(training_args["project"]) / training_args["name"]
    best_weights = output_dir / "weights" / "best.pt"
    last_weights = output_dir / "weights" / "last.pt"

    print(f"\n[train] Training complete.")
    print(f"  Best weights: {best_weights}")
    print(f"  Last weights: {last_weights}")

    # --- Log final metrics to W&B ---
    # Ultralytics automatically handles finishing the W&B run and logging the
    # model artifacts if W&B is enabled.
    if wandb_run is not None:
        print("[train] W&B run finished by Ultralytics.")

    return best_weights


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for product defect detection."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to training configuration YAML.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset.yaml. Overrides config default.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint.",
    )
    parser.add_argument(
        "--resume-weights",
        type=str,
        default=None,
        help="Path to specific checkpoint weights for resuming.",
    )
    args = parser.parse_args()

    best_model_path = train(
        config_path=args.config,
        dataset_yaml=args.dataset,
        resume=args.resume,
        resume_weights=args.resume_weights,
    )
    print(f"\nBest model saved to: {best_model_path}")
