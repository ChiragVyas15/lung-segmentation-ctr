import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(input_tensor, num_filters):
    """Dual Convolution Block: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU."""
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_unet(input_shape=(512, 512, 1), num_classes=1):
    """
    Standard U-Net Architecture implemented in TensorFlow/Keras.
    
    Encoder: 4 contracting blocks with MaxPooling.
    Bottleneck: Dual convolution block at smallest spatial dimensions.
    Decoder: 4 expanding blocks with UpSampling2D + Skip Connection Concatenation.
    Final output: 1x1 Conv2D with Sigmoid activation.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # --- ENCODER (Contracting Path) ---
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # --- BOTTLENECK ---
    b = conv_block(p4, 512)

    # --- DECODER (Expanding Path) ---
    u4 = layers.UpSampling2D((2, 2))(b)
    concat4 = layers.Concatenate()([u4, c4])
    c5 = conv_block(concat4, 256)

    u3 = layers.UpSampling2D((2, 2))(c5)
    concat3 = layers.Concatenate()([u3, c3])
    c6 = conv_block(concat3, 128)

    u2 = layers.UpSampling2D((2, 2))(c6)
    concat2 = layers.Concatenate()([u2, c2])
    c7 = conv_block(concat2, 64)

    u1 = layers.UpSampling2D((2, 2))(c7)
    concat1 = layers.Concatenate()([u1, c1])
    c8 = conv_block(concat1, 32)

    # --- FINAL OUTPUT ---
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", name="unet_output")(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name="U-Net")
    return model
