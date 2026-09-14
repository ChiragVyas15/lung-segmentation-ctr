import tensorflow as tf
from tensorflow.keras import layers, models

def squeeze_excite_block(input_tensor, ratio=8):
    """Squeeze and Excitation (SE) block for feature recalibration."""
    init = input_tensor
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = layers.Multiply()([init, se])
    return x

def stem_block(input_tensor, num_filters):
    """Stem Block of ResU-Net++."""
    x = layers.Conv2D(num_filters, 3, padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)

    shortcut = layers.Conv2D(num_filters, 1, padding='same')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = squeeze_excite_block(x)
    return x

def resnet_block(input_tensor, num_filters, stride=1):
    """Residual Block with Squeeze and Excitation."""
    x = layers.BatchNormalization()(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters, 3, padding='same')(x)

    shortcut = layers.Conv2D(num_filters, 1, strides=stride, padding='same')(input_tensor)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = squeeze_excite_block(x)
    return x

def aspp_block(input_tensor, num_filters):
    """Atrous Spatial Pyramid Pooling (ASPP) Block."""
    x1 = layers.Conv2D(num_filters, 1, padding='same')(input_tensor)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(num_filters, 3, dilation_rate=6, padding='same')(input_tensor)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)

    x3 = layers.Conv2D(num_filters, 3, dilation_rate=12, padding='same')(input_tensor)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Activation('relu')(x3)

    x4 = layers.Conv2D(num_filters, 3, dilation_rate=18, padding='same')(input_tensor)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Activation('relu')(x4)

    x = layers.Concatenate()([x1, x2, x3, x4])
    x = layers.Conv2D(num_filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_resunetplusplus(input_shape=(512, 512, 1), num_classes=1):
    """
    ResU-Net++ Architecture implemented strictly using TensorFlow/Keras layers.
    
    Includes stem block, residual blocks with Squeeze & Excitation,
    ASPP multi-scale feature bottleneck, and progressive skip connection decoder.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Encoder
    x1 = stem_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(x1)

    x2 = resnet_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(x2)

    x3 = resnet_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(x3)

    x4 = resnet_block(p3, 256)
    p4 = layers.MaxPooling2D((2, 2))(x4)

    # Bottleneck ASPP
    b = aspp_block(p4, 512)

    # Decoder
    d4 = layers.UpSampling2D((2, 2))(b)
    d4 = layers.Concatenate()([d4, x4])
    d4 = resnet_block(d4, 256)

    d3 = layers.UpSampling2D((2, 2))(d4)
    d3 = layers.Concatenate()([d3, x3])
    d3 = resnet_block(d3, 128)

    d2 = layers.UpSampling2D((2, 2))(d3)
    d2 = layers.Concatenate()([d2, x2])
    d2 = resnet_block(d2, 64)

    d1 = layers.UpSampling2D((2, 2))(d2)
    d1 = layers.Concatenate()([d1, x1])
    d1 = resnet_block(d1, 32)

    # Final Output
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid', name="resunetpp_output")(d1)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResUNetPlusPlus")
    return model
