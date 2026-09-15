import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.preprocessing import load_and_preprocess_image, load_and_preprocess_mask
from src.postprocessing import postprocess_mask
from src.metrics import calculate_segmentation_metrics
from src.ctr import calculate_ctr_geometry
from src.visualization import plot_ctr_visualization, plot_actual_vs_predicted

# Streamlit Page Config
st.set_page_config(
    page_title="Lung Segmentation & CTR Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        font-weight: 400;
        color: #475569;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title & Header
st.markdown('<div class="main-title">Lung Segmentation & Cardiothoracic Ratio Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Comparison of SegNet, U-Net, ResU-Net++ and Attention U-Net (TensorFlow/Keras)</div>', unsafe_allow_html=True)

# Medical Disclaimer Banner
st.warning("⚠️ **Medical Safety Disclaimer**: This system is intended for research and educational purposes only and is not a medical diagnostic device.")

# Model Caching
@st.cache_resource
def load_all_segmentation_models():
    """Loads and caches all four trained TensorFlow/Keras models."""
    models = {}
    model_keys = ["segnet", "unet", "resunetpp", "attunet"]
    
    for key in model_keys:
        path = config.MODEL_PATHS[key]
        if path.exists():
            models[key] = tf.keras.models.load_model(str(path), compile=False)
        else:
            models[key] = None
    return models

models_dict = load_all_segmentation_models()

# Sidebar Controls
st.sidebar.title("🛠️ Configuration & Controls")
threshold = st.sidebar.slider("Segmentation Threshold", 0.1, 0.9, float(config.THRESHOLD), 0.05)
min_area = st.sidebar.slider("Min Component Area (px)", 50, 2000, int(config.MIN_COMPONENT_AREA), 50)

# Sidebar Navigation Tabs
app_mode = st.sidebar.radio("Navigate", [
    "🚀 Image Inference & CTR Analysis",
    "🖼️ Dataset Test Gallery",
    "📊 Overall Model Performance Benchmark"
])

# Utility Helper Functions
def process_single_image(img_input, models, gt_mask_path=None):
    """Runs all 4 models on a single image and computes segmentation + CTR outputs."""
    if isinstance(img_input, (str, Path)):
        img = load_and_preprocess_image(str(img_input))
    else:
        file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
        decoded = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(decoded, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]))
        img = (resized.astype(np.float32) / 255.0)[:, :, np.newaxis]

    img_batch = np.expand_dims(img, axis=0)
    
    gt_mask = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_mask = load_and_preprocess_mask(gt_mask_path)

    results = {}
    model_keys = ["segnet", "unet", "resunetpp", "attunet"]
    model_names = ["SegNet", "U-Net", "ResU-Net++", "AttU-Net"]

    for key, name in zip(model_keys, model_names):
        model = models.get(key)
        if model is None:
            results[key] = {
                'name': name,
                'mask': np.zeros_like(img),
                'ctr': np.nan,
                'cardiac_dia': np.nan,
                'thoracic_dia': np.nan,
                'points': None,
                'metrics': None
            }
            continue

        raw_pred = model.predict(img_batch, verbose=0)[0]
        clean_mask_2d = postprocess_mask(raw_pred, threshold=threshold, min_area=min_area)
        clean_mask_3d = np.expand_dims(clean_mask_2d, axis=-1)

        ctr_val, cardiac_dia, thoracic_dia, points, reason = calculate_ctr_geometry(clean_mask_2d)

        metrics = None
        if gt_mask is not None:
            metrics = calculate_segmentation_metrics(gt_mask, clean_mask_3d, threshold=threshold)

        results[key] = {
            'name': name,
            'mask': clean_mask_3d,
            'ctr': ctr_val,
            'cardiac_dia': cardiac_dia,
            'thoracic_dia': thoracic_dia,
            'points': points,
            'metrics': metrics
        }

    return img, gt_mask, results

# TAB 1: IMAGE INFERENCE & CTR ANALYSIS
if app_mode == "🚀 Image Inference & CTR Analysis":
    st.header("Select or Upload a Chest X-Ray Image")

    # Discover built-in sample images in sample_images folder
    sample_dir = BASE_DIR / "sample_images"
    sample_files = sorted(list(sample_dir.glob("*_0.png"))) if sample_dir.exists() else []

    input_source = st.radio("Choose Input Method:", ["🖼️ Select from Pre-loaded Samples", "📤 Upload Custom X-Ray Image"], horizontal=True)

    selected_img_path = None
    selected_mask_path = None
    uploaded_file = None

    if input_source == "🖼️ Select from Pre-loaded Samples" and sample_files:
        sample_choice = st.selectbox("Choose a Sample Chest X-Ray:", [f.name for f in sample_files])
        selected_img_path = sample_dir / sample_choice
        mask_name = sample_choice.replace(".png", "_mask.png")
        if (sample_dir / mask_name).exists():
            selected_mask_path = sample_dir / mask_name
    elif input_source == "📤 Upload Custom X-Ray Image":
        uploaded_file = st.file_uploader("Choose a Chest X-Ray image file (PNG / JPG)", type=["png", "jpg", "jpeg"])

    active_input = selected_img_path or uploaded_file

    if active_input is not None:
        st.subheader("Original Image & Ground Truth")
        col_img, col_gt = st.columns(2)
        
        img, gt_mask, results = process_single_image(active_input, models_dict, gt_mask_path=selected_mask_path)

        with col_img:
            st.image((img.squeeze() * 255).astype(np.uint8), caption="Input Chest X-Ray", use_container_width=True)
        
        with col_gt:
            if gt_mask is not None:
                st.image(gt_mask.squeeze(), caption="Ground Truth Lung Mask", use_container_width=True)
            else:
                st.info("Ground truth mask unavailable for custom uploaded image.")

        st.markdown("---")
        st.header("Multi-Model Lung Segmentation & CTR Geometric Diagrams")

        cols = st.columns(4)
        for idx, (key, res) in enumerate(results.items()):
            with cols[idx]:
                st.subheader(res['name'])
                st.image(res['mask'].squeeze(), caption=f"{res['name']} Mask", use_container_width=True)
                
                # Annotated CTR Diagram
                fig_ctr = plot_ctr_visualization(img, res['mask'], res['points'], res['ctr'], 
                                                res['cardiac_dia'], res['thoracic_dia'], model_name=res['name'])
                st.pyplot(fig_ctr)

                if not np.isnan(res['ctr']):
                    st.metric(f"{res['name']} CTR", f"{res['ctr']:.4f}")
                    st.caption(f"Cardiac: {res['cardiac_dia']:.1f} px | Thoracic: {res['thoracic_dia']:.1f} px")
                else:
                    st.error("Invalid Geometry")

        st.markdown("---")
        st.header("Multi-Model CTR & Segmentation Metrics Comparison")
        
        table_data = []
        for key, res in results.items():
            m = res['metrics']
            row_info = {
                'Model': res['name'],
                'CTR': f"{res['ctr']:.4f}" if not np.isnan(res['ctr']) else "NaN",
                'Cardiac Dia (px)': f"{res['cardiac_dia']:.1f}" if not np.isnan(res['cardiac_dia']) else "NaN",
                'Thoracic Dia (px)': f"{res['thoracic_dia']:.1f}" if not np.isnan(res['thoracic_dia']) else "NaN"
            }
            if m:
                row_info.update({
                    'Dice Score': f"{m['dice']:.4f}",
                    'IoU': f"{m['iou']:.4f}",
                    'Accuracy': f"{m['accuracy']:.4f}",
                    'HD95 (px)': f"{m['hd95']:.2f}"
                })
            table_data.append(row_info)
            
        st.table(pd.DataFrame(table_data))
        st.info("ℹ️ **CTR Note**: CTR values are calculated geometrically without heart segmentation following the research paper methodology.")

# TAB 2: DATASET TEST GALLERY
elif app_mode == "🖼️ Dataset Test Gallery":
    st.header("Dataset Labelled Test Set Gallery")
    split_file = config.OUTPUTS_DIR / "dataset_split.csv"
    
    if split_file.exists():
        split_df = pd.read_csv(split_file)
        test_df = split_df[split_df['split'] == 'test'].reset_index(drop=True)
        
        selected_file = st.selectbox("Select a test image from dataset:", test_df['image_filename'].values)
        row = test_df[test_df['image_filename'] == selected_file].iloc[0]
        
        img, gt_mask, results = process_single_image(row['image_path'], models_dict, gt_mask_path=row['mask_path'])

        st.subheader("Actual vs Predicted Mask Comparison")
        model_masks = [res['mask'] for res in results.values()]
        model_names = [res['name'] for res in results.values()]
        fig_comp = plot_actual_vs_predicted(img, gt_mask, model_masks, model_names)
        st.pyplot(fig_comp)

        st.subheader("Segmentation Performance Metrics for Selected Test Image")
        metric_table = []
        for key, res in results.items():
            m = res['metrics']
            if m:
                metric_table.append({
                    'Model': res['name'],
                    'Pixel Accuracy': f"{m['accuracy']:.4f}",
                    'Precision': f"{m['precision']:.4f}",
                    'Recall': f"{m['recall']:.4f}",
                    'Specificity': f"{m['specificity']:.4f}",
                    'Dice Score': f"{m['dice']:.4f}",
                    'IoU': f"{m['iou']:.4f}",
                    'HD95 (px)': f"{m['hd95']:.2f}"
                })
        if metric_table:
            st.table(pd.DataFrame(metric_table))
        else:
            st.warning("Ground-truth segmentation mask unavailable for this image.")
    else:
        st.error("Dataset split file not found. Run training/evaluation first.")

# TAB 3: OVERALL PERFORMANCE BENCHMARK
elif app_mode == "📊 Overall Model Performance Benchmark":
    st.header("Overall Held-Out Test Set Performance Benchmark")
    res_file = config.RESULTS_DIR / "segmentation_results.csv"

    if res_file.exists():
        df_res = pd.read_csv(res_file)
        st.subheader("Aggregated Evaluation Table (15% Test Split)")
        st.dataframe(df_res.style.highlight_max(subset=['dice', 'iou', 'accuracy'], color='#D1FAE5')
                                 .highlight_min(subset=['hd95'], color='#FEF3C7'))

        best_dice_row = df_res.loc[df_res['dice'].idxmax()]
        best_iou_row = df_res.loc[df_res['iou'].idxmax()]
        best_hd95_row = df_res.loc[df_res['hd95'].idxmin()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🥇 Best Dice Score", f"{best_dice_row['model']}", f"{best_dice_row['dice']:.4f}")
        with col2:
            st.metric("🥇 Best IoU", f"{best_iou_row['model']}", f"{best_iou_row['iou']:.4f}")
        with col3:
            st.metric("🥇 Lowest HD95", f"{best_hd95_row['model']}", f"{best_hd95_row['hd95']:.2f} px")

        st.subheader("Metric Comparison Charts")
        plot_files = list(config.PLOTS_DIR.glob("comparison_*.png"))
        if plot_files:
            for p in plot_files:
                st.image(str(p), caption=p.stem, use_container_width=True)
    else:
        st.info("Evaluation results not found yet. Run `evaluation/evaluate_segmentation.py` to populate.")

# Footer
st.markdown("---")
st.caption("Lung Segmentation and Automatic CTR Calculation System | TensorFlow 2.x & Streamlit")
