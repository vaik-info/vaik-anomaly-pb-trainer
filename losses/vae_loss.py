import tensorflow as tf


def mse_loss(y_true, y_pred):
    r_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=[1, 2, 3])
    return 1000 * r_loss


def kl_loss(mean, log_var):
    return -0.5 * tf.keras.backend.sum(1 + log_var - tf.keras.backend.square(mean) - tf.keras.backend.exp(log_var),
                                       axis=1)


def vae_loss(y_true, y_pred, mean, var):
    return mse_loss(tf.cast(y_true, tf.float32) / 255., y_pred) + kl_loss(mean, var)
