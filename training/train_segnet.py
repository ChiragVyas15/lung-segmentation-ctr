import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from models_arch.segnet import build_segnet
from training.train_utils import train_single_model

if __name__ == "__main__":
    train_single_model("segnet", build_segnet, config.MODEL_PATHS["segnet"])
