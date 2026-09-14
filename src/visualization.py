import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.lung_analysis import extract_lung_contours

def plot_ctr_visualization(image, mask, points, ctr, cardiac_dia, thoracic_dia, model_name="Model", save_path=None):
    """
    Plots the annotated CTR geometric diagram showing:
    - Original X-ray with lung contours overlaid
    - Points D2', D3', E1, E2
    - Thoracic diameter line (D2' to D3') in Cyan
    - Cardiac diameter line (E1 to E2) in Magenta
    - Annotated CTR value
    """
    img_display = (image.squeeze() * 255).astype(np.uint8)
    if len(img_display.shape) == 2:
        img_rgb = cv2.cvtColor(img_display, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_display.copy()

    mask_binary = (mask.squeeze() > 0.5).astype(np.uint8)
    cnt_right, cnt_left = extract_lung_contours(mask_binary)

    # Draw contours (Green)
    if cnt_right is not None:
        cv2.drawContours(img_rgb, [cnt_right], -1, (0, 255, 0), 2)
    if cnt_left is not None:
        cv2.drawContours(img_rgb, [cnt_left], -1, (0, 255, 0), 2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_rgb)

    if points is not None:
        d2 = points['D2_prime']
        d3 = points['D3_prime']
        e1 = points['E1']
        e2 = points['E2']

        # Draw Thoracic Line (Cyan)
        ax.plot([d2[0], d3[0]], [d2[1], d3[1]], color='cyan', linewidth=2.5, label=f'Thoracic Dia (D2-D3): {thoracic_dia:.1f} px')
        ax.scatter([d2[0], d3[0]], [d2[1], d3[1]], color='cyan', s=60, zorder=5)
        ax.text(d2[0]-15, d2[1]-10, "D2'", color='cyan', fontweight='bold', fontsize=12)
        ax.text(d3[0]+5, d3[1]-10, "D3'", color='cyan', fontweight='bold', fontsize=12)

        # Draw Cardiac Line (Magenta)
        ax.plot([e1[0], e2[0]], [e1[1], e2[1]], color='magenta', linewidth=2.5, label=f'Cardiac Dia (E1-E2): {cardiac_dia:.1f} px')
        ax.scatter([e1[0], e2[0]], [e1[1], e2[1]], color='magenta', s=60, zorder=5)
        ax.text(e1[0]-15, e1[1]+15, "E1", color='magenta', fontweight='bold', fontsize=12)
        ax.text(e2[0]+5, e2[1]+15, "E2", color='magenta', fontweight='bold', fontsize=12)

    title_text = f"{model_name} CTR: {ctr:.4f}" if not np.isnan(ctr) else f"{model_name} CTR: Invalid Geometry"
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    return fig

def plot_actual_vs_predicted(image, gt_mask, model_masks, model_names, save_path=None):
    """
    Generates a 2x3 comparison panel showing:
    1. Original X-Ray
    2. Ground Truth Mask
    3. SegNet Prediction
    4. U-Net Prediction
    5. ResU-Net++ Prediction
    6. AttU-Net Prediction
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(image.squeeze(), cmap='gray')
    axes[0].set_title("Original X-Ray", fontweight='bold')
    axes[0].axis('off')

    if gt_mask is not None:
        axes[1].imshow(gt_mask.squeeze(), cmap='gray')
        axes[1].set_title("Ground Truth Mask", fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, "Ground Truth\nUnavailable", ha='center', va='center', fontsize=12)
        axes[1].set_title("Ground Truth Mask", fontweight='bold')
    axes[1].axis('off')

    for idx, (mask, name) in enumerate(zip(model_masks, model_names)):
        axes[idx + 2].imshow(mask.squeeze(), cmap='gray')
        axes[idx + 2].set_title(f"{name} Prediction", fontweight='bold')
        axes[idx + 2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    return fig

def plot_training_curves(history_df, model_name, save_path=None):
    """Plots training and validation Loss, Dice, and Accuracy curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history_df['loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history_df.columns:
        axes[0].plot(history_df['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title(f"{model_name} - Loss", fontweight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice Coefficient
    if 'dice_coef' in history_df.columns:
        axes[1].plot(history_df['dice_coef'], label='Train Dice', linewidth=2)
        if 'val_dice_coef' in history_df.columns:
            axes[1].plot(history_df['val_dice_coef'], label='Val Dice', linewidth=2)
        axes[1].set_title(f"{model_name} - Dice Coefficient", fontweight='bold')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Dice Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Accuracy
    acc_col = 'accuracy' if 'accuracy' in history_df.columns else 'pixel_accuracy'
    if acc_col in history_df.columns:
        axes[2].plot(history_df[acc_col], label='Train Accuracy', linewidth=2)
        if f'val_{acc_col}' in history_df.columns:
            axes[2].plot(history_df[f'val_{acc_col}'], label='Val Accuracy', linewidth=2)
        axes[2].set_title(f"{model_name} - Pixel Accuracy", fontweight='bold')
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    return fig

def plot_model_comparison_bar_charts(results_df, save_dir=config.PLOTS_DIR):
    """Generates individual comparison bar charts for all 7 segmentation metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'dice', 'iou', 'hd95']
    models = results_df['model'].values

    for metric in metrics:
        if metric in results_df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            values = results_df[metric].values
            bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=0.5)

            ax.set_title(f"Model Comparison - {metric.upper()}", fontsize=14, fontweight='bold')
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1.0)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.4f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plt.savefig(save_dir / f"comparison_{metric}.png", bbox_inches='tight', dpi=150)
            plt.close()
