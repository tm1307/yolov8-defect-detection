
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import yaml
from pycocotools.coco import COCO
from tqdm import tqdm

COCO_TRAIN_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_TRAIN_ANN_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)

YOLO_DATASET_YAML_TEMPLATE = """
path: {dataset_root}
train: images/train
val: images/val

nc: {num_classes}
names: {class_names}

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary containing all configuration parameters.

    All transforms are bounding-box safe — they adjust bbox coordinates
    automatically. We use the 'yolo' format (normalized center x, y, w, h).

    Args:
        config: Dictionary containing augmentation parameters under the
                'augmentation' key.

    Returns:
        An Albumentations Compose object configured for YOLO-format bboxes.

    Args:
        coco_api: A pycocotools COCO object loaded with annotations.
        target_categories: List of COCO category names to include.

    Returns:
        Dictionary mapping original COCO category ID → new 0-indexed class ID.

    Raises:
        ValueError: If any target category is not found in COCO.

    YOLO format: [center_x, center_y, width, height] normalized to [0, 1].

    Args:
        bbox: COCO-format bounding box [x_min, y_min, w, h] in pixels.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.

    Returns:
        List of 4 floats in YOLO format [cx, cy, w, h], all in [0, 1].

    This function:
    1. Loads the COCO annotation JSON
    2. Finds images containing at least one target category
    3. Copies those images to the output directory
    4. Writes YOLO-format label .txt files alongside the images

    Args:
        annotation_file: Path to COCO annotation JSON (e.g., instances_train2017.json).
        images_dir: Path to the directory containing COCO images.
        output_images_dir: Destination directory for filtered images.
        output_labels_dir: Destination directory for YOLO label files.
        category_mapping: Optional pre-computed {coco_cat_id: yolo_class_id} mapping.
            If None, will be computed from target_categories.
        target_categories: List of COCO category names. Required if
            category_mapping is None.
        max_images: Maximum number of images to include (for Colab feasibility).
        seed: Random seed for reproducible subset selection.

    Returns:
        Number of images successfully processed.

    Raises:
        ValueError: If neither category_mapping nor target_categories is provided.

    For each original image, generates `num_augmented_per_image` augmented
    copies with transformed bounding boxes. Originals are also copied.

    Args:
        images_dir: Source directory of training images.
        labels_dir: Source directory of YOLO-format label files.
        output_images_dir: Destination for augmented images.
        output_labels_dir: Destination for augmented labels.
        augmentation_pipeline: Albumentations Compose with bbox support.
        num_augmented_per_image: How many augmented copies per original image.

    Returns:
        Total number of images in the augmented dataset (originals + augmented).

    Args:
        dataset_root: Absolute path to the dataset root directory.
        class_names: Ordered list of class names (index = class ID).
        output_path: Where to write the YAML. Defaults to dataset_root/dataset.yaml.

    Returns:
        Path to the generated YAML file.

    This is the main entry point for data preparation. It orchestrates:
    1. Loading config
    2. Filtering COCO to target categories
    3. Converting to YOLO format
    4. Applying augmentations to training split
    5. Generating the dataset.yaml for Ultralytics

    NOTE: This function assumes COCO images and annotations have already been
    downloaded to the paths specified in config. For Colab, use the provided
    notebook which handles downloading via shell commands (much faster than
    Python HTTP for multi-GB files).

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Path to the generated dataset.yaml file.
