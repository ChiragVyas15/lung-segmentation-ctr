import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from models_arch.segnet import build_segnet
from models_arch.unet import build_unet
from models_arch.resunetpp import build_resunetplusplus
from models_arch.attunet import build_attention_unet
from training.train_utils import train_single_model

def train_all_models():
    """Trains all four lung segmentation models sequentially."""
    print("Training SegNet...")
    train_single_model("segnet", build_segnet, config.MODEL_PATHS["segnet"])

    print("Training U-Net...")
    train_single_model("unet", build_unet, config.MODEL_PATHS["unet"])

    print("Training ResU-Net++...")
    train_single_model("resunetpp", build_resunetplusplus, config.MODEL_PATHS["resunetpp"])

    print("Training AttU-Net...")
    train_single_model("attunet", build_attention_unet, config.MODEL_PATHS["attunet"])

    print("\n=======================================================")
    print("      All four models trained successfully.")
    print("=======================================================\n")

if __name__ == "__main__":
    train_all_models()
