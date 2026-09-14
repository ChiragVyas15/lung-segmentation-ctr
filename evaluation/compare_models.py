import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.visualization import plot_model_comparison_bar_charts

def generate_comparison_plots():
    """Generates comparison bar charts from outputs/results/segmentation_results.csv."""
    results_path = config.RESULTS_DIR / "segmentation_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}. Run evaluation first!")

    results_df = pd.read_csv(results_path)
    print(f"[COMPARISON] Generating bar charts from {results_path}...")
    plot_model_comparison_bar_charts(results_df, save_dir=config.PLOTS_DIR)
    print(f"[INFO] Comparison plots saved to {config.PLOTS_DIR}")

if __name__ == "__main__":
    generate_comparison_plots()
