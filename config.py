import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_ROOT = BASE_DIR / "Lung Segmentation"
if not DATASET_ROOT.exists():
    DATASET_ROOT = BASE_DIR / "chexmask datset and model" / "Lung Segmentation"
if not DATASET_ROOT.exists():
    DATASET_ROOT = BASE_DIR / "data" / "Lung Segmentation"

CXR_DIR = DATASET_ROOT / "CXR_png"
MASKS_DIR = DATASET_ROOT / "masks"
TEST_DIR = DATASET_ROOT / "test"

MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

RESULTS_DIR = OUTPUTS_DIR / "results"
PLOTS_DIR = OUTPUTS_DIR / "plots"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
CTR_VISUALIZATIONS_DIR = OUTPUTS_DIR / "ctr_visualizations"
HISTORY_DIR = OUTPUTS_DIR / "history"
DATASET_REPORT_DIR = OUTPUTS_DIR / "dataset_report"

# Ensure all directories exist
for d in [MODELS_DIR, OUTPUTS_DIR, RESULTS_DIR, PLOTS_DIR, VISUALIZATIONS_DIR, 
          CTR_VISUALIZATIONS_DIR, HISTORY_DIR, DATASET_REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Centralized Parameters (Configurable Image Size)
IMAGE_SIZE = (256, 256)
INPUT_SHAPE = (256, 256, 1)
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
THRESHOLD = 0.5
RANDOM_SEED = 42

# Post-processing Parameters
MIN_COMPONENT_AREA = 200

# Model Names & Target Save Paths
MODEL_PATHS = {
    "segnet": MODELS_DIR / "segnet_best.keras",
    "unet": MODELS_DIR / "unet_best.keras",
    "resunetpp": MODELS_DIR / "resunetpp_best.keras",
    "attunet": MODELS_DIR / "attunet_best.keras"
}
