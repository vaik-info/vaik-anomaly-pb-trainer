import tensorflow as tf
import numpy as np


def branch_decoder(encoder_branch, output_shape, pool_num=3, filters=64):
    x = encoder_branch
    for i in range(pool_num):
        x = tf.keras.layers.Conv2DTranspose(filters=filters*2, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # conv-block-4
    outputs = tf.keras.layers.Conv2DTranspose(filters=output_shape[-1], kernel_size=3, strides=1, padding='same',
                                              activation='sigmoid')(x)
    return outputs