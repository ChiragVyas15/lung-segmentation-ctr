import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.postprocessing import get_left_lung, get_right_lung, postprocess_mask

def extract_lung_contours(binary_mask):
    """
    Extracts contours from a binary lung mask.
    Returns left_lung_contour, right_lung_contour, and full_contours list.
    """
    clean_mask = postprocess_mask(binary_mask) if binary_mask.dtype != np.uint8 else binary_mask
    right_mask = get_right_lung(clean_mask)
    left_mask = get_left_lung(clean_mask)

    contours_right, _ = cv2.findContours(right_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_left, _ = cv2.findContours(left_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt_right = contours_right[0] if len(contours_right) > 0 else None
    cnt_left = contours_left[0] if len(contours_left) > 0 else None

    return cnt_right, cnt_left

def extract_boundary_extrema(contour):
    """
    Determines upper, lower, medial, and lateral extrema points for a single lung contour.
    """
    if contour is None or len(contour) == 0:
        return None

    pts = contour.squeeze()
    if pts.ndim != 2:
        return None

    top_pt = pts[np.argmin(pts[:, 1])]     # Min y
    bottom_pt = pts[np.argmax(pts[:, 1])]  # Max y
    left_pt = pts[np.argmin(pts[:, 0])]    # Min x
    right_pt = pts[np.argmax(pts[:, 0])]   # Max x

    return {
        'upper': top_pt,
        'lower': bottom_pt,
        'leftmost': left_pt,
        'rightmost': right_pt
    }
