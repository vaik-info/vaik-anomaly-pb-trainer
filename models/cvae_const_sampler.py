import tensorflow as tf


def __sampling_const_reparameterization(distribution_params):
    mean, log_var = distribution_params
    z = mean + tf.keras.backend.exp(log_var / 2)
    return z


def prepare(mean_input, log_var_input):
    z = tf.keras.layers.Lambda(__sampling_const_reparameterization)([mean_input, log_var_input])
    model = tf.keras.Model([mean_input, log_var_input], z)
    return model, z
