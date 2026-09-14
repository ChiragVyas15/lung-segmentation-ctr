import sys
import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import label

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config

def postprocess_mask(prob_mask, threshold=config.THRESHOLD, min_area=config.MIN_COMPONENT_AREA):
    """
    Applies thresholding and connected-component analysis to clean raw model output.
    Removes small isolated noise components and retains major lung regions.
    """
    # 1. Threshold probability map
    binary = (prob_mask.squeeze() > threshold).astype(np.uint8)

    # 2. Connected Component Labeling using OpenCV
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(binary, dtype=np.uint8)

    # Filter background (label 0)
    valid_labels = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_labels.append((i, area, centroids[i][0]))  # (label, area, centroid_x)

    # Sort components by area descending and pick top 2 (Left and Right lungs)
    valid_labels.sort(key=lambda x: x[1], reverse=True)
    top_labels = [item[0] for item in valid_labels[:2]]

    # Reconstruct cleaned binary mask
    clean_mask = np.isin(labels, top_labels).astype(np.uint8)
    return clean_mask

def get_lung_mask(prob_mask, threshold=config.THRESHOLD, min_area=config.MIN_COMPONENT_AREA):
    """Returns the cleaned 2D binary lung mask with channel dimension (H, W, 1)."""
    clean_2d = postprocess_mask(prob_mask, threshold, min_area)
    return np.expand_dims(clean_2d, axis=-1)

def get_left_lung(prob_mask, threshold=config.THRESHOLD, min_area=config.MIN_COMPONENT_AREA):
    """
    Extracts the patient's LEFT lung mask.
    Note: Radiologically, the patient's left lung appears on the RIGHT side of the X-ray (centroid x > image_center_x).
    """
    clean_mask = postprocess_mask(prob_mask, threshold, min_area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(clean_mask, dtype=np.uint8)

    img_width = clean_mask.shape[1]
    center_x = img_width / 2.0

    left_lung_label = None
    max_x = -1

    for i in range(1, num_labels):
        c_x = centroids[i][0]
        # Patient's left lung is on the right side of image
        if c_x >= center_x and c_x > max_x:
            max_x = c_x
            left_lung_label = i

    if left_lung_label is None:
        # Fallback to rightmost component if none crossed center
        left_lung_label = np.argmax([centroids[i][0] for i in range(1, num_labels)]) + 1

    return (labels == left_lung_label).astype(np.uint8)

def get_right_lung(prob_mask, threshold=config.THRESHOLD, min_area=config.MIN_COMPONENT_AREA):
    """
    Extracts the patient's RIGHT lung mask.
    Note: Radiologically, the patient's right lung appears on the LEFT side of the X-ray (centroid x < image_center_x).
    """
    clean_mask = postprocess_mask(prob_mask, threshold, min_area)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean_mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(clean_mask, dtype=np.uint8)

    img_width = clean_mask.shape[1]
    center_x = img_width / 2.0

    right_lung_label = None
    min_x = img_width + 1

    for i in range(1, num_labels):
        c_x = centroids[i][0]
        # Patient's right lung is on the left side of image
        if c_x < center_x and c_x < min_x:
            min_x = c_x
            right_lung_label = i

    if right_lung_label is None:
        # Fallback to leftmost component
        right_lung_label = np.argmin([centroids[i][0] for i in range(1, num_labels)]) + 1

    return (labels == right_lung_label).astype(np.uint8)
