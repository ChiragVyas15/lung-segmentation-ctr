---
title: Heart CTR & Lung Segmentation
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# Lung Segmentation and Automatic Cardiothoracic Ratio (CTR) Calculation

> **Research Paper Reference**: *"Automatic cardiothoracic ratio calculation based on lung fields abstracted from chest X-ray images without heart segmentation"*

---

## 📌 Project Overview
This repository provides a complete, runnable, end-to-end deep learning solution for **automatic lung field segmentation** and **cardiothoracic ratio (CTR) calculation** from Chest X-Ray (CXR) images.

The core methodology abstracts the lung fields to extract key anatomical boundary points ($D_2'$, $D_3'$, $E_1$, $E_2$) and compute the thoracic and cardiac transverse diameters **without requiring a separate heart segmentation model**.

---

## ⚠️ Key Disclaimers & Technical Mandates

1. **TensorFlow/Keras Exclusive**:
   - The entire codebase is implemented **100% in TensorFlow 2.x and Keras** (`tf.keras.layers`).
   - Strictly zero PyTorch dependencies or imports (`torch`, `torchvision`, `.pth`, `.pt`, etc.).

2. **Qualitative Test Set Usage**:
   - The approximately 96 unlabelled test images are used for **qualitative inference only**. Quantitative metrics (Dice, IoU, Accuracy, HD95) are calculated strictly on the held-out labelled test set (~106 paired images).

3. **CTR Ground Truth Limitation**:
   - CTR ground truth is not provided by the Kaggle dataset. CTR accuracy, MAE, or RMSE are **not fabricated**. The reported CTR is the model-derived geometric estimate. If an external `ctr_ground_truth.csv` is provided, MAE, RMSE, and Pearson correlation are computed automatically.

4. **Medical Safety Disclaimer**:
   - *This system is intended for research and educational purposes only and is not a medical diagnostic device.*

---

## 🧠 Four Segmentation Models
All four deep learning architectures are implemented in `models_arch/` using `tf.keras`:

1. **SegNet** (`models_arch/segnet.py`): Encoder-decoder structure with convolutional blocks, batch normalization, and progressive upsampling reconstruction.
2. **U-Net** (`models_arch/unet.py`): Standard U-Net contracting and expanding paths with skip connections.
3. **ResU-Net++** (`models_arch/resunetpp.py`): Residual building blocks, Squeeze-and-Excitation (SE) recalibration, ASPP multi-scale feature bottleneck, and progressive skip connection decoding.
4. **Attention U-Net / AttU-Net** (`models_arch/attunet.py`): Integrates Attention Gates (AGs) into skip connections to suppress irrelevant background features and emphasize lung region representations.

---

## 📏 CTR Geometric Methodology
Following the paper methodology:
- **Connected Component Processing**: Removes small isolated background noise regions and isolates the left and right lung fields.
- **Thoracic Transverse Points ($D_2'$, $D_3'$)**: Leftmost boundary point of right lung and rightmost boundary point of left lung at the widest thoracic width.
- **Cardiac Transverse Points ($E_1$, $E_2$)**: Medial-most points of right and left lung boundaries facing the cardiac shadow.
- **Cardiac Diameter**: $\Delta x_1 = |x_{E1} - x_{E2}|$
- **Thoracic Diameter**: $\Delta x_2 = |x_{D2'} - x_{D3'}|$
- **Cardiothoracic Ratio**: $CTR = \frac{\Delta x_1}{\Delta x_2}$

---

## 📁 Repository Structure

```
.
├── app.py                            # Streamlit web application
├── config.py                         # Centralized configuration
├── requirements.txt                  # TF/Keras dependency list
├── README.md                         # Documentation
├── lung_segmentation_ctr.ipynb       # 31-Section Jupyter Notebook
│
├── models/                           # Best Keras models
│   ├── segnet_best.keras
│   ├── unet_best.keras
│   ├── resunetpp_best.keras
│   └── attunet_best.keras
│
├── models_arch/                      # Keras architecture definitions
│   ├── segnet.py
│   ├── unet.py
│   ├── resunetpp.py
│   └── attunet.py
│
├── src/                              # Core utility modules
│   ├── dataset.py                    # Dataset discovery & splitting
│   ├── preprocessing.py              # Grayscale conversion & sizing
│   ├── augmentation.py               # Synchronized tf.data pipeline
│   ├── metrics.py                    # Dice, IoU, Accuracy, HD95
│   ├── postprocessing.py             # Connected-component filtering
│   ├── lung_analysis.py              # Contour extraction
│   ├── ctr.py                        # Geometric CTR calculation
│   └── visualization.py              # Plotting & diagrams
│
├── training/                         # Model training scripts
│   ├── train_segnet.py
│   ├── train_unet.py
│   ├── train_resunetpp.py
│   ├── train_attunet.py
│   └── train_all.py                  # Sequential training script
│
├── evaluation/                       # Evaluation & model comparison
│   ├── evaluate_segmentation.py      # Benchmark evaluation
│   ├── evaluate_ctr.py              # CTR calculation script
│   └── compare_models.py             # Plotting generator
│
└── outputs/                          # Generated outputs
    ├── dataset_report/
    ├── history/
    ├── plots/
    ├── results/
    ├── visualizations/
    └── ctr_visualizations/
```

---

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation & Splitting
```bash
python src/dataset.py
```

### 3. Model Training
Train all four models sequentially:
```bash
python training/train_all.py
```

### 4. Benchmark Evaluation
```bash
python evaluation/evaluate_segmentation.py
python evaluation/evaluate_ctr.py
python evaluation/compare_models.py
```

### 5. Launch Streamlit Web App
```bash
streamlit run app.py
```
