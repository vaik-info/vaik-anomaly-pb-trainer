import tensorflow as tf


def __sampling_reparameterization(distribution_params):
    mean, log_var = distribution_params
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean), mean=0., stddev=1.)
    z = mean + tf.keras.backend.exp(log_var / 2) * epsilon
    return z


def prepare(mean_input, log_var_input):
    z = tf.keras.layers.Lambda(__sampling_reparameterization)([mean_input, log_var_input])
    model = tf.keras.Model([mean_input, log_var_input], z)
    return model, z
