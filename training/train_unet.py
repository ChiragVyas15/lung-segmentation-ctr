import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from models_arch.unet import build_unet
from training.train_utils import train_single_model

if __name__ == "__main__":
    train_single_model("unet", build_unet, config.MODEL_PATHS["unet"])
