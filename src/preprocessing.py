import os
import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

def load_and_preprocess_image(image_path, target_size=config.IMAGE_SIZE):
    """
    Loads a chest X-ray image, converts to grayscale, resizes to target_size,
    casts to float32, and normalizes pixel values to [0, 1].
    Returns array of shape (H, W, 1).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image using OpenCV
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to decode image at path: {image_path}")

    # Resize to target_size (Width, Height)
    img_resized = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1] float32
    img_normalized = img_resized.astype(np.float32) / 255.0

    # Add channel dimension (H, W, 1)
    if len(img_normalized.shape) == 2:
        img_normalized = np.expand_dims(img_normalized, axis=-1)

    return img_normalized

def load_and_preprocess_mask(mask_path, target_size=config.IMAGE_SIZE):
    """
    Loads a binary lung mask, resizes using NEAREST NEIGHBOR interpolation,
    thresholds to strict binary values {0, 1}, and returns array of shape (H, W, 1).
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at path: {mask_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to decode mask at path: {mask_path}")

    # RESIZE USING NEAREST NEIGHBOR INTERPOLATION ONLY (Mandatory prompt constraint)
    mask_resized = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)

    # Threshold to strict binary 0 or 1
    binary_mask = (mask_resized > 127).astype(np.float32)

    # Add channel dimension (H, W, 1)
    if len(binary_mask.shape) == 2:
        binary_mask = np.expand_dims(binary_mask, axis=-1)

    return binary_mask
