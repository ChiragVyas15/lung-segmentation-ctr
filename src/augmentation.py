import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.preprocessing import load_and_preprocess_image, load_and_preprocess_mask

def augment_pair(image, mask):
    """
    Applies synchronized spatial augmentations to image and mask,
    and intensity augmentations to image only.
    """
    # 1. Random Horizontal Flip
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)

    # 2. Random Brightness & Contrast (Image only)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.clip_by_value(image, 0.0, 1.0)

    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask

def create_tf_dataset(df, batch_size=config.BATCH_SIZE, is_training=True, shuffle=True):
    """
    Constructs an efficient tf.data.Dataset pipeline from dataframe containing image_path & mask_path.
    """
    img_paths = df['image_path'].values
    mask_paths = df['mask_path'].values

    def py_load_fn(img_p, mask_p):
        img_p_str = img_p.numpy().decode('utf-8')
        mask_p_str = mask_p.numpy().decode('utf-8')
        img = load_and_preprocess_image(img_p_str)
        mask = load_and_preprocess_mask(mask_p_str)
        return img.astype(np.float32), mask.astype(np.float32)

    def tf_load_fn(img_p, mask_p):
        img, mask = tf.py_function(
            func=py_load_fn,
            inp=[img_p, mask_p],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape(config.INPUT_SHAPE)
        mask.set_shape(config.INPUT_SHAPE)
        return img, mask

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=config.RANDOM_SEED)

    ds = ds.map(tf_load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
