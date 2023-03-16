import tensorflow as tf
import numpy as np


def decoder(z, shape_before_flattening, output_shape, filters=32):
    x = tf.keras.layers.Dense(np.prod(shape_before_flattening[1:]))(z)
    x = tf.keras.layers.Reshape(shape_before_flattening[1:])(x)

    # conv-block-1
    x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # conv-block-2-3
    for i in range(4):
        x = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # conv-block-4
    outputs = tf.keras.layers.Conv2DTranspose(filters=output_shape[-1], kernel_size=3, strides=1, padding='same',
                                              activation='sigmoid')(x)
    model = tf.keras.Model(z, outputs)
    return model, outputs
