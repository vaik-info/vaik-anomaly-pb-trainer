import tensorflow as tf


def __calc_mse(distribution_params):
    model_input, model_outputs = distribution_params
    return tf.math.squared_difference(model_outputs, tf.cast(model_input, tf.float32) / 255.)


def prepare(model_input, model_outputs):
    output_mse = tf.keras.layers.Lambda(__calc_mse)([model_input, model_outputs])
    model = tf.keras.Model(model_input, output_mse)
    return model
