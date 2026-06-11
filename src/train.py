
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_wandb(config: dict[str, Any]) -> Any | None:
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
        "seed": config["data"]["seed"],
        "deterministic": True,
        "amp": True,
        "cache": False,
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
    config = load_config(config_path)
    print(f"[train] Loaded config from: {config_path}")

    env_info = validate_environment()

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

    wandb_run = setup_wandb(config)
    if wandb_run is not None:
        wandb_run.log({"environment": env_info})

    model_architecture = config["model"]["architecture"]
    if resume and resume_weights:
        print(f"[train] Resuming from: {resume_weights}")
        model = YOLO(resume_weights)
    else:
        print(f"[train] Loading pretrained: {model_architecture}")
        model = YOLO(model_architecture)

    training_args = build_training_args(config, dataset_yaml)
    print("[train] Training arguments:")
    for key, value in sorted(training_args.items()):
        print(f"  {key}: {value}")

    print("\n[train] Starting training...")
    results = model.train(**training_args)

    output_dir = Path(training_args["project"]) / training_args["name"]
    best_weights = output_dir / "weights" / "best.pt"
    last_weights = output_dir / "weights" / "last.pt"

    print(f"\n[train] Training complete.")
    print(f"  Best weights: {best_weights}")
    print(f"  Last weights: {last_weights}")

    if wandb_run is not None:
        print("[train] W&B run finished by Ultralytics.")

    return best_weights

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
