"""HayabusaNet model definition.

This module contains the reusable TensorFlow/Keras implementation of
HayabusaNet. It is intended to be imported by notebooks or scripts that need
to build the model architecture for training, evaluation, or post-hoc
visualization such as Grad-CAM and Grad-CAM++.

No training code is included in this module.
"""

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.layers import (
    Activation,
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
)


@tf.keras.utils.register_keras_serializable(package="HayabusaNet")
class ChannelAttention(layers.Layer):
    """Channel attention module from CBAM.

    This layer recalibrates feature channels using shared MLP outputs from
    global average pooling and global max pooling descriptors.
    """

    def __init__(self, reduction_ratio=16, **kwargs):

        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):

        channels = int(input_shape[-1])
        hidden_units = max(channels // self.reduction_ratio, 1)

        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.gmp = layers.GlobalMaxPooling2D(keepdims=True)
        self.fc1 = layers.Dense(hidden_units, activation="relu")
        self.fc2 = layers.Dense(channels)

    def call(self, x):

        x_dtype = x.dtype

        avg = self.gap(x)
        maxp = self.gmp(x)

        avg_mlp = self.fc2(self.fc1(tf.cast(avg, tf.float32)))
        max_mlp = self.fc2(self.fc1(tf.cast(maxp, tf.float32)))

        channel_attention = tf.nn.sigmoid(avg_mlp + max_mlp)
        channel_attention = tf.cast(channel_attention, x_dtype)

        return x * channel_attention

    def get_config(self):

        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


@tf.keras.utils.register_keras_serializable(package="HayabusaNet")
class SpatialAttention(layers.Layer):
    """Spatial attention module from CBAM.

    This layer emphasizes informative spatial regions using channel-wise
    average and max pooling followed by a convolutional attention map.
    """

    def __init__(self, kernel_size=7, **kwargs):

        super().__init__(**kwargs)
        if kernel_size not in (3, 5, 7):
            raise ValueError("Common CBAM kernel sizes are 3, 5, or 7.")

        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="sigmoid",
        )

    def call(self, x):
 
        x_dtype = x.dtype

        avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        maxp = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg, maxp], axis=-1)

        spatial_attention = self.conv(tf.cast(concat, tf.float32))
        spatial_attention = tf.cast(spatial_attention, x_dtype)

        return x * spatial_attention

    def get_config(self):
 
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable(package="HayabusaNet")
class CBAMBlock(layers.Layer):
    """Convolutional Block Attention Module.

    This block applies channel attention followed by spatial attention.
    """

    def __init__(self, reduction_ratio=16, spatial_kernel=7, **kwargs):

        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel = spatial_kernel
        self.ca = ChannelAttention(reduction_ratio=reduction_ratio)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def call(self, x):

        x = self.ca(x)
        x = self.sa(x)
        return x

    def get_config(self):

        config = super().get_config()
        config.update(
            {
                "reduction_ratio": self.reduction_ratio,
                "spatial_kernel": self.spatial_kernel,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="HayabusaNet")
class SelfAttention(layers.Layer):
    """Lightweight self-attention module.

    This layer models global contextual dependencies using query, key, and
    value projections on a downsampled feature map.
    """

    def __init__(self, in_dim, **kwargs):
 
        super().__init__(**kwargs)
        self.in_dim = int(in_dim)

        projection_dim = max(self.in_dim // 8, 1)
        self.query_conv = layers.Conv2D(projection_dim, kernel_size=1)
        self.key_conv = layers.Conv2D(projection_dim, kernel_size=1)
        self.value_conv = layers.Conv2D(self.in_dim, kernel_size=1)

        self.gamma = self.add_weight(
            name="gamma",
            shape=(),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):

        input_dtype = x.dtype

        # Downsampling keeps the self-attention operation lightweight for
        # high-resolution feature maps.
        x_down = tf.nn.avg_pool2d(x, ksize=4, strides=4, padding="SAME")
        x_down_float32 = tf.cast(x_down, tf.float32)

        q = self.query_conv(x_down_float32)
        k = self.key_conv(x_down_float32)
        v = self.value_conv(x_down_float32)

        projection_dim = max(self.in_dim // 8, 1)
        q = tf.reshape(q, [tf.shape(x_down)[0], -1, projection_dim])
        k = tf.reshape(k, [tf.shape(x_down)[0], -1, projection_dim])
        v = tf.reshape(v, [tf.shape(x_down)[0], -1, self.in_dim])

        energy = tf.matmul(q, k, transpose_b=True)
        attention = tf.nn.softmax(energy)
        out = tf.matmul(attention, v)
        out = tf.reshape(out, tf.shape(x_down))

        out = tf.image.resize(out, tf.shape(x)[1:3])
        out = tf.cast(out, input_dtype)
        gamma = tf.cast(self.gamma, input_dtype)

        return gamma * out + x

    def get_config(self):

        config = super().get_config()
        config.update({"in_dim": self.in_dim})
        return config


@tf.keras.utils.register_keras_serializable(package="HayabusaNet")
class HybridAttention(layers.Layer):
    """Hybrid attention module used in HayabusaNet.

    This layer combines CBAM-based local refinement and self-attention-based
    global context using residual-gated fusion.
    """

    def __init__(self, in_dim, alpha=0.5, **kwargs):

        super().__init__(**kwargs)
        self.in_dim = int(in_dim)
        self.alpha_init = float(alpha)

        self.cbam = CBAMBlock()
        self.self_attn = SelfAttention(self.in_dim)

        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.alpha_init),
            trainable=True,
            dtype=tf.float32,
        )

    def call(self, x):

        cbam_out = self.cbam(x)
        sa_out = self.self_attn(x)

        x_dtype = x.dtype
        cbam_out = tf.cast(cbam_out, x_dtype)
        sa_out = tf.cast(sa_out, x_dtype)
        alpha = tf.cast(self.alpha, x_dtype)

        return x + alpha * (cbam_out + sa_out)

    def get_config(self):

        config = super().get_config()
        config.update({"in_dim": self.in_dim, "alpha": self.alpha_init})
        return config


def branch(x, k, depth, base_filters=32):
    """Build one multiscale convolutional branch.

    Args:
        x: Input feature map.
        k: Kernel size used in the separable convolution layers.
        depth: Number of repeated convolution-pooling blocks.
        base_filters: Number of filters in the first convolutional block.

    Returns:
        Output feature map from one multiscale branch.
    """
    c = x

    for i in range(depth):
        filters = base_filters * (2**i)
        c = layers.SeparableConv2D(
            filters,
            (k, k),
            padding="same",
            strides=1,
            use_bias=False,
        )(c)
        c = layers.Activation("relu")(c)
        c = layers.MaxPooling2D(2, padding="same")(c)

    return c


def build_hayabusanet(
    input_shape=(224, 224, 3),
    num_classes=4,
    model_name="HayabusaNet_Multiclass",
):
    """Build the HayabusaNet architecture.

    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes.
        model_name: Name assigned to the Keras model.

    Returns:
        HayabusaNet Keras model.
    """
    input_layer = Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255)(input_layer)

    x1 = branch(x, k=3, depth=5)
    x2 = branch(x, k=5, depth=5)
    x3 = branch(x, k=7, depth=5)

    fused = Add()([x1, x2, x3])
    fused = HybridAttention(in_dim=fused.shape[-1])(fused)

    gap = GlobalAveragePooling2D()(fused)

    fc = Dense(512, kernel_regularizer=regularizers.l2(1e-4))(gap)
    fc = Activation("relu")(fc)
    fc = Dropout(0.3)(fc)

    fc = Dense(256, kernel_regularizer=regularizers.l2(1e-4))(fc)
    fc = Activation("relu")(fc)
    fc = Dropout(0.3)(fc)

    output_layer = Dense(num_classes, activation="softmax", dtype="float32")(fc)

    return Model(inputs=input_layer, outputs=output_layer, name=model_name)
