import sys
import pandas as pd
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.utils import set_seed
from src.dataset import create_dataset_split
from src.augmentation import create_tf_dataset
from src.metrics import dice_coef, dice_loss, bce_dice_loss
from src.visualization import plot_training_curves

def train_single_model(model_name, build_fn, save_path):
    """
    Standardized training routine for a segmentation model:
    - Sets seed=42
    - Loads train & val datasets
    - Compiles model with Adam (lr=1e-4), BCE loss, dice_coef metric
    - Sets up ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    - Fits model and saves best checkpoint (.keras)
    - Exports training history CSV and training curve plot
    """
    set_seed(config.RANDOM_SEED)
    print(f"\n=======================================================")
    print(f"       STARTING TRAINING FOR: {model_name.upper()}")
    print(f"=======================================================")

    split_df = create_dataset_split()
    train_df = split_df[split_df['split'] == 'train']
    val_df = split_df[split_df['split'] == 'val']

    train_ds = create_tf_dataset(train_df, batch_size=config.BATCH_SIZE, is_training=True, shuffle=True)
    val_ds = create_tf_dataset(val_df, batch_size=config.BATCH_SIZE, is_training=False, shuffle=False)

    model = build_fn(input_shape=config.INPUT_SHAPE, num_classes=1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    loss = tf.keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[dice_coef, 'accuracy']
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(save_path),
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coef',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coef',
            mode='max',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history to CSV
    hist_df = pd.DataFrame(history.history)
    history_csv = config.HISTORY_DIR / f"{model_name}_history.csv"
    hist_df.to_csv(history_csv, index=False)
    print(f"[INFO] History saved to {history_csv}")

    # Plot and save training curves
    plot_path = config.PLOTS_DIR / f"{model_name}_training.png"
    plot_training_curves(hist_df, model_name=model_name.upper(), save_path=plot_path)
    print(f"[INFO] Training plot saved to {plot_path}")

    print(f"[SUCCESS] {model_name.upper()} best model saved to {save_path}\n")
    return model, hist_df
