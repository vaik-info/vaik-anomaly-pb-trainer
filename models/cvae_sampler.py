import tensorflow as tf


def __sampling_reparameterization(distribution_params):
    mean, log_var = distribution_params
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean), mean=0., stddev=1.)
    z = mean + tf.keras.backend.exp(log_var / 2) * epsilon
    return z


def prepare(mean_input, log_var_input):
    mean = tf.keras.Input(shape=tf.keras.backend.int_shape(mean_input)[-1])
    log_var = tf.keras.Input(shape=tf.keras.backend.int_shape(log_var_input)[-1])
    out = tf.keras.layers.Lambda(__sampling_reparameterization)([mean, log_var])
    model = tf.keras.Model([mean, log_var], out)
    return model
