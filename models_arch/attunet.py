import tensorflow as tf
from tensorflow.keras import layers, models

def attention_gate(x, g, inter_channels):
    """
    Attention Gate (AG) mechanism for Attention U-Net.
    
    x: Skip connection signal from encoder.
    g: Gating signal from lower decoder layer.
    inter_channels: Intermediate filter size.
    """
    # Gating signal transform
    theta_x = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(x)
    phi_g = layers.Conv2D(inter_channels, 1, strides=1, padding='same')(g)

    # Combine signals
    f = layers.Activation('relu')(layers.Add()([theta_x, phi_g]))
    
    # Generate attention coefficients (alpha)
    psi_f = layers.Conv2D(1, 1, strides=1, padding='same')(f)
    rate = layers.Activation('sigmoid')(psi_f)

    # Multiply skip features with attention weights
    out = layers.Multiply()([x, rate])
    return out

def conv_block(input_tensor, num_filters):
    """Dual Convolution Block."""
    x = layers.Conv2D(num_filters, 3, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_attention_unet(input_shape=(512, 512, 1), num_classes=1):
    """
    Attention U-Net (AttU-Net) Architecture implemented in TensorFlow/Keras.
    
    Uses Attention Gates on skip connections to suppress irrelevant background
    features and highlight lung fields.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # --- ENCODER ---
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

    # --- DECODER WITH ATTENTION GATES ---
    u4 = layers.UpSampling2D((2, 2))(b)
    g4 = attention_gate(x=c4, g=u4, inter_channels=128)
    concat4 = layers.Concatenate()([u4, g4])
    c5 = conv_block(concat4, 256)

    u3 = layers.UpSampling2D((2, 2))(c5)
    g3 = attention_gate(x=c3, g=u3, inter_channels=64)
    concat3 = layers.Concatenate()([u3, g3])
    c6 = conv_block(concat3, 128)

    u2 = layers.UpSampling2D((2, 2))(c6)
    g2 = attention_gate(x=c2, g=u2, inter_channels=32)
    concat2 = layers.Concatenate()([u2, g2])
    c7 = conv_block(concat2, 64)

    u1 = layers.UpSampling2D((2, 2))(c7)
    g1 = attention_gate(x=c1, g=u1, inter_channels=16)
    concat1 = layers.Concatenate()([u1, g1])
    c8 = conv_block(concat1, 32)

    # --- FINAL OUTPUT ---
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", name="attunet_output")(c8)

    model = models.Model(inputs=inputs, outputs=outputs, name="AttentionUNet")
    return model
