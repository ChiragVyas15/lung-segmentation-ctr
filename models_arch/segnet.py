import tensorflow as tf
from tensorflow.keras import layers, models

def build_segnet(input_shape=(512, 512, 1), num_classes=1):
    """
    SegNet Architecture implemented in TensorFlow/Keras.
    
    Encoder: Progressive Conv2D + BatchNorm + ReLU + MaxPooling2D blocks.
    Decoder: Progressive UpSampling2D + Conv2D + BatchNorm + ReLU blocks.
    Final output: 1x1 Conv2D with Sigmoid activation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # --- ENCODER ---
    # Encoder Block 1
    x = layers.Conv2D(64, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Encoder Block 2
    x = layers.Conv2D(128, 3, padding="same")(p1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Encoder Block 3
    x = layers.Conv2D(256, 3, padding="same")(p2)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Encoder Block 4 (Bottleneck)
    x = layers.Conv2D(512, 3, padding="same")(p3)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # --- DECODER ---
    # Decoder Block 1
    u1 = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(256, 3, padding="same")(u1)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Decoder Block 2
    u2 = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, 3, padding="same")(u2)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Decoder Block 3
    u3 = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding="same")(u3)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Final Output Layer
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", name="segnet_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="SegNet")
    return model
