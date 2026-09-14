import sys
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Keras Loss & Metrics for Model Training
@tf.function
def dice_coef(y_true, y_pred, smooth=1e-6):
    """Computes Dice Coefficient for TF/Keras tensors."""
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), tf.float32)
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@tf.function
def dice_loss(y_true, y_pred):
    """Computes Dice Loss for TF/Keras tensors."""
    return 1.0 - dice_coef(y_true, y_pred)

@tf.function
def bce_dice_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy + Dice Loss."""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# Evaluation Metrics for Binary Segmentation Arrays
def calculate_segmentation_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculates 7 comprehensive segmentation evaluation metrics for binary numpy arrays:
    1. Pixel Accuracy
    2. Precision
    3. Recall / Sensitivity
    4. Specificity
    5. Dice Score
    6. IoU
    7. HD95 (95th percentile Hausdorff Distance)
    """
    y_true_b = (y_true > 0.5).astype(np.uint8).flatten()
    y_pred_b = (y_pred > threshold).astype(np.uint8).flatten()

    tp = np.sum((y_true_b == 1) & (y_pred_b == 1))
    tn = np.sum((y_true_b == 0) & (y_pred_b == 0))
    fp = np.sum((y_true_b == 0) & (y_pred_b == 1))
    fn = np.sum((y_true_b == 1) & (y_pred_b == 0))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    dice = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    hd95 = compute_hd95(y_true.squeeze(), y_pred.squeeze(), threshold=threshold)

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'dice': float(dice),
        'iou': float(iou),
        'hd95': float(hd95)
    }

def compute_hd95(mask_true, mask_pred, threshold=0.5):
    """
    Computes the 95th percentile Hausdorff Distance (HD95) using vectorized distance matrices.
    """
    gt_binary = (mask_true > 0.5).astype(np.uint8)
    pred_binary = (mask_pred > threshold).astype(np.uint8)

    if np.sum(gt_binary) == 0 or np.sum(pred_binary) == 0:
        return 0.0

    contours_gt, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_pred, _ = cv2.findContours(pred_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours_gt or not contours_pred:
        return 0.0

    pts_gt = np.vstack([c.squeeze() for c in contours_gt if c.squeeze().ndim == 2])
    pts_pred = np.vstack([c.squeeze() for c in contours_pred if c.squeeze().ndim == 2])

    if len(pts_gt) == 0 or len(pts_pred) == 0:
        return 0.0

    # Vectorized Distance Matrix computation using scipy.spatial.distance.cdist
    dist_matrix = cdist(pts_gt, pts_pred, metric='euclidean')
    d_gt_pred = np.min(dist_matrix, axis=1)
    d_pred_gt = np.min(dist_matrix, axis=0)

    all_distances = np.concatenate([d_gt_pred, d_pred_gt])
    hd95_val = np.percentile(all_distances, 95)

    return float(hd95_val)
