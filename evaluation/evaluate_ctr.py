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
from src.preprocessing import load_and_preprocess_image
from src.postprocessing import postprocess_mask
from src.ctr import calculate_ctr_geometry
from src.visualization import plot_ctr_visualization

def evaluate_ctr():
    """
    Evaluates CTR geometry calculation independently for all four segmentation models using batched inference.
    Saves outputs/results/ctr_results.csv and annotated CTR diagrams in outputs/ctr_visualizations/.
    """
    print("[CTR EVALUATION] Loading test split images...")
    split_df = pd.read_csv(config.OUTPUTS_DIR / "dataset_split.csv")
    test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)

    loaded_models = {}
    model_keys = ["segnet", "unet", "resunetpp", "attunet"]
    model_names = ["SegNet", "U-Net", "ResU-Net++", "AttU-Net"]

    for key, path in config.MODEL_PATHS.items():
        loaded_models[key] = tf.keras.models.load_model(str(path), compile=False)

    print("\n[CTR EVALUATION] Pre-loading test image batch...")
    all_imgs = np.array([load_and_preprocess_image(p) for p in test_df['image_path']])

    predictions = {}
    for key in model_keys:
        print(f"[CTR EVALUATION] Batched inference for {key.upper()}...")
        predictions[key] = loaded_models[key].predict(all_imgs, batch_size=16, verbose=1)

    ctr_records = []

    print("\n[CTR EVALUATION] Computing per-model CTR and geometric diameters...")
    for idx in tqdm(range(len(test_df)), desc="CTR Calculation"):
        img_id = test_df.iloc[idx]['image_filename']
        img = all_imgs[idx]

        record = {'image_id': img_id}

        for key, name in zip(model_keys, model_names):
            raw_pred = predictions[key][idx]
            clean_mask = postprocess_mask(raw_pred)

            ctr_val, cardiac_dia, thoracic_dia, points, reason = calculate_ctr_geometry(clean_mask)

            record[f'{key}_ctr'] = ctr_val
            record[f'{key}_cardiac_diameter'] = cardiac_dia
            record[f'{key}_thoracic_diameter'] = thoracic_dia

            # Save sample CTR visualization for first 5 test images
            if idx < 5:
                viz_path = config.CTR_VISUALIZATIONS_DIR / f"{Path(img_id).stem}_{key}_ctr.png"
                plot_ctr_visualization(img, clean_mask, points, ctr_val, cardiac_dia, thoracic_dia, 
                                       model_name=name, save_path=viz_path)

        ctr_records.append(record)

    ctr_df = pd.DataFrame(ctr_records)
    out_csv = config.RESULTS_DIR / "ctr_results.csv"
    ctr_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] CTR evaluation results saved to {out_csv}")

    print("\n------------------------------------------------------------------------------------------")
    print(" [NOTE ON CTR GROUND TRUTH LIMITATION]")
    print(" CTR ground truth is not available in this dataset. The displayed CTR is the model-derived estimate.")
    print(" If ctr_ground_truth.csv is provided later, MAE/RMSE/Correlation will be automatically calculated.")
    print("------------------------------------------------------------------------------------------\n")

    return ctr_df

if __name__ == "__main__":
    evaluate_ctr()
