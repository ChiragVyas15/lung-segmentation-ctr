import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from sklearn.model_selection import GroupShuffleSplit

def discover_dataset():
    """
    Scans the dataset directories, matches X-rays to ground truth masks,
    identifies unlabelled test images, and generates a dataset summary report.
    """
    cxr_dir = Path(config.CXR_DIR)
    masks_dir = Path(config.MASKS_DIR)
    test_dir = Path(config.TEST_DIR)

    all_cxr = sorted([f for f in os.listdir(cxr_dir) if f.endswith('.png')]) if cxr_dir.exists() else []
    all_masks = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')]) if masks_dir.exists() else []
    all_test = sorted([f for f in os.listdir(test_dir) if f.endswith('.png')]) if test_dir.exists() else []

    paired_data = []
    unmatched_cxr = []
    
    # Map mask filename without '_mask' to actual mask path
    mask_dict = {}
    for m in all_masks:
        base_name = m.replace('_mask.png', '').replace('.png', '')
        mask_dict[base_name] = os.path.join(masks_dir, m)

    for img_file in all_cxr:
        stem = Path(img_file).stem
        if stem in mask_dict:
            img_path = os.path.join(cxr_dir, img_file)
            mask_path = mask_dict[stem]
            patient_id = stem.rsplit('_', 1)[0] if '_' in stem else stem
            paired_data.append({
                'image_filename': img_file,
                'mask_filename': os.path.basename(mask_path),
                'image_path': img_path,
                'mask_path': mask_path,
                'patient_id': patient_id
            })
        else:
            unmatched_cxr.append(img_file)

    report = {
        'total_xrays': len(all_cxr),
        'total_masks': len(all_masks),
        'matched_pairs': len(paired_data),
        'unmatched_xrays': len(unmatched_cxr),
        'unlabelled_test_images': len(all_test)
    }

    report_df = pd.DataFrame([report])
    report_df.to_csv(config.DATASET_REPORT_DIR / "dataset_summary.csv", index=False)
    
    print("[DATASET REPORT]")
    for k, v in report.items():
        print(f"  {k}: {v}")

    return pd.DataFrame(paired_data), all_test, report

def create_dataset_split(force_recreate=False):
    """
    Creates a 70% Train / 15% Val / 15% Test split grouped by patient_id
    to prevent data leakage across splits.
    Saves split to outputs/dataset_split.csv.
    """
    split_file = config.OUTPUTS_DIR / "dataset_split.csv"
    if split_file.exists() and not force_recreate:
        print(f"[INFO] Loading existing dataset split from {split_file}")
        return pd.read_csv(split_file)

    df_pairs, _, _ = discover_dataset()
    if df_pairs.empty:
        raise ValueError(f"No matched image-mask pairs found at {config.CXR_DIR}!")

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=config.RANDOM_SEED)
    train_idx, temp_idx = next(gss1.split(df_pairs, groups=df_pairs['patient_id']))

    train_df = df_pairs.iloc[train_idx].copy()
    temp_df = df_pairs.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=config.RANDOM_SEED)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_df, groups=temp_df['patient_id']))

    val_df = temp_df.iloc[val_sub_idx].copy()
    test_df = temp_df.iloc[test_sub_idx].copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_split_df.to_csv(split_file, index=False)

    print(f"\n[DATASET SPLIT CREATED] Saved to {split_file}")
    print(f"  Train images: {len(train_df)} ({len(train_df['patient_id'].unique())} patients)")
    print(f"  Val images:   {len(val_df)} ({len(val_df['patient_id'].unique())} patients)")
    print(f"  Test images:  {len(test_df)} ({len(test_df['patient_id'].unique())} patients)")

    return full_split_df

if __name__ == "__main__":
    create_dataset_split(force_recreate=True)
