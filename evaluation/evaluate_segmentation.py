import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.preprocessing import load_and_preprocess_image, load_and_preprocess_mask
from src.postprocessing import postprocess_mask
from src.metrics import calculate_segmentation_metrics
from src.visualization import plot_actual_vs_predicted

def evaluate_segmentation():
    """
    Evaluates all four trained segmentation models on the held-out test dataset using batched inference.
    Calculates Pixel Accuracy, Precision, Recall, Specificity, Dice, IoU, and HD95 per image.
    Generates summary tables, CSV outputs, and actual-vs-predicted visual comparisons.
    """
    print("[EVALUATION] Loading test dataset split...")
    split_df = pd.read_csv(config.OUTPUTS_DIR / "dataset_split.csv")
    test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)

    if test_df.empty:
        raise ValueError("Test dataset split is empty!")

    print(f"[EVALUATION] Found {len(test_df)} test samples with ground truth masks.")

    loaded_models = {}
    model_keys = ["segnet", "unet", "resunetpp", "attunet"]
    model_names_list = ["SegNet", "U-Net", "ResU-Net++", "AttU-Net"]

    for name in model_keys:
        path = config.MODEL_PATHS[name]
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}. Run training first!")
        print(f"[EVALUATION] Loading {name.upper()} model from {path}...")
        loaded_models[name] = tf.keras.models.load_model(str(path), compile=False)

    # Pre-load all test images and masks into batch tensors for fast inference
    print("\n[EVALUATION] Pre-loading test image batch...")
    all_imgs = np.array([load_and_preprocess_image(p) for p in test_df['image_path']])
    all_gt_masks = [load_and_preprocess_mask(p) for p in test_df['mask_path']]

    # Batched model predictions
    predictions = {}
    for key in model_keys:
        print(f"[EVALUATION] Batched inference for {key.upper()}...")
        predictions[key] = loaded_models[key].predict(all_imgs, batch_size=16, verbose=1)

    per_image_results = []

    print("\n[EVALUATION] Post-processing and calculating metrics...")
    for idx in tqdm(range(len(test_df)), desc="Calculating Metrics"):
        img_id = test_df.iloc[idx]['image_filename']
        img = all_imgs[idx]
        gt_mask = all_gt_masks[idx]

        model_masks_list = []

        for key in model_keys:
            raw_pred = predictions[key][idx]
            clean_mask = postprocess_mask(raw_pred)
            clean_mask_3d = np.expand_dims(clean_mask, axis=-1)

            metrics = calculate_segmentation_metrics(gt_mask, clean_mask_3d)

            row_dict = {'image_id': img_id, 'model': key}
            row_dict.update(metrics)
            per_image_results.append(row_dict)

            model_masks_list.append(clean_mask_3d)

        # Save actual vs predicted visualization for first 5 test samples
        if idx < 5:
            viz_path = config.VISUALIZATIONS_DIR / f"{Path(img_id).stem}_comparison.png"
            plot_actual_vs_predicted(img, gt_mask, model_masks_list, model_names_list, save_path=viz_path)

    # Save per-image results dataframe
    per_img_df = pd.DataFrame(per_image_results)
    per_img_df.to_csv(config.RESULTS_DIR / "segmentation_per_image.csv", index=False)
    print(f"\n[INFO] Per-image results saved to {config.RESULTS_DIR / 'segmentation_per_image.csv'}")

    # Aggregated Summary Table (Mean ± Std)
    summary_rows = []
    for key, display_name in zip(model_keys, model_names_list):
        sub_df = per_img_df[per_img_df['model'] == key]
        summary_rows.append({
            'model': display_name,
            'accuracy': sub_df['accuracy'].mean(),
            'accuracy_std': sub_df['accuracy'].std(),
            'precision': sub_df['precision'].mean(),
            'precision_std': sub_df['precision'].std(),
            'recall': sub_df['recall'].mean(),
            'recall_std': sub_df['recall'].std(),
            'specificity': sub_df['specificity'].mean(),
            'specificity_std': sub_df['specificity'].std(),
            'dice': sub_df['dice'].mean(),
            'dice_std': sub_df['dice'].std(),
            'iou': sub_df['iou'].mean(),
            'iou_std': sub_df['iou'].std(),
            'hd95': sub_df['hd95'].mean(),
            'hd95_std': sub_df['hd95'].std()
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(config.RESULTS_DIR / "segmentation_results.csv", index=False)
    print(f"[INFO] Summary segmentation metrics saved to {config.RESULTS_DIR / 'segmentation_results.csv'}\n")

    print("==========================================================================================")
    print("                              FINAL SEGMENTATION BENCHMARK RESULTS")
    print("==========================================================================================")
    print(summary_df[['model', 'accuracy', 'precision', 'recall', 'specificity', 'dice', 'iou', 'hd95']].to_string(index=False))
    print("==========================================================================================\n")

    return summary_df

if __name__ == "__main__":
    evaluate_segmentation()
