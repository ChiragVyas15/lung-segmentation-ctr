import sys
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.postprocessing import postprocess_mask, get_right_lung, get_left_lung
from src.lung_analysis import extract_lung_contours

def calculate_ctr_geometry(binary_mask):
    """
    Calculates the Cardiothoracic Ratio (CTR) based on lung field geometry
    following the research paper methodology:
    - Points D2' and D3': Thoracic transverse outer-edge points (widest thoracic width).
    - Points E1 and E2: Cardiac border points (medial-most points facing the heart).
    
    Returns:
        ctr (float or NaN)
        cardiac_diameter (float or NaN)
        thoracic_diameter (float or NaN)
        points (dict containing D2_prime, D3_prime, E1, E2)
        reason (str explaining any failure)
    """
    clean_mask = postprocess_mask(binary_mask) if binary_mask.dtype != np.uint8 else binary_mask
    cnt_right, cnt_left = extract_lung_contours(clean_mask)

    if cnt_right is None or cnt_left is None:
        return np.nan, np.nan, np.nan, None, "Failed to isolate both left and right lung contours."

    pts_right = cnt_right.squeeze()
    pts_left = cnt_left.squeeze()

    if pts_right.ndim != 2 or pts_left.ndim != 2 or len(pts_right) < 5 or len(pts_left) < 5:
        return np.nan, np.nan, np.nan, None, "Insufficient contour boundary points."

    # 1. Thoracic Transverse Diameter Points (D2' and D3')
    # D2' is the leftmost point on the right lung (min x in right lung contour)
    # D3' is the rightmost point on the left lung (max x in left lung contour)
    idx_d2 = np.argmin(pts_right[:, 0])
    D2_prime = pts_right[idx_d2]  # (x, y)

    idx_d3 = np.argmax(pts_left[:, 0])
    D3_prime = pts_left[idx_d3]   # (x, y)

    thoracic_diameter = abs(float(D3_prime[0]) - float(D2_prime[0]))

    # 2. Cardiac Transverse Diameter Points (E1 and E2)
    # E1 is the medial-most point of the right lung (max x on right lung contour facing cardiac silhouette)
    # E2 is the medial-most point of the left lung (min x on left lung contour facing cardiac silhouette)
    # Focus on lower 60% of vertical span where cardiac silhouette borders the lungs
    min_y_r, max_y_r = np.min(pts_right[:, 1]), np.max(pts_right[:, 1])
    min_y_l, max_y_l = np.min(pts_left[:, 1]), np.max(pts_left[:, 1])

    lower_region_r = pts_right[pts_right[:, 1] >= min_y_r + 0.3 * (max_y_r - min_y_r)]
    lower_region_l = pts_left[pts_left[:, 1] >= min_y_l + 0.3 * (max_y_l - min_y_l)]

    if len(lower_region_r) == 0 or len(lower_region_l) == 0:
        lower_region_r = pts_right
        lower_region_l = pts_left

    idx_e1 = np.argmax(lower_region_r[:, 0])
    E1 = lower_region_r[idx_e1]

    idx_e2 = np.argmin(lower_region_l[:, 0])
    E2 = lower_region_l[idx_e2]

    cardiac_diameter = abs(float(E2[0]) - float(E1[0]))

    if thoracic_diameter <= 0:
        return np.nan, np.nan, np.nan, None, "Thoracic diameter zero or negative."

    ctr = cardiac_diameter / thoracic_diameter

    # Sanity check for valid CTR bounds (typically between 0.35 and 0.85 for human X-rays)
    if ctr < 0.20 or ctr > 0.95:
        reason = f"CTR value {ctr:.3f} outside expected physiological bounds (0.20 - 0.95)."
    else:
        reason = "Success"

    points = {
        'D2_prime': tuple(map(int, D2_prime)),
        'D3_prime': tuple(map(int, D3_prime)),
        'E1': tuple(map(int, E1)),
        'E2': tuple(map(int, E2))
    }

    return float(ctr), float(cardiac_diameter), float(thoracic_diameter), points, reason
