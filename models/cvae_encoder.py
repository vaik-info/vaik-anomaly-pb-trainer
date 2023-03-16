import tensorflow as tf

def prepare(input_shape=(224, 224, 1), filters=32, latent_dim=2):
    inputs = tf.keras.Input(shape=input_shape)
    rescale_inputs = tf.keras.layers.Rescaling(scale=1. / 255)(inputs)

    # conv-block-1
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(rescale_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # conv-block-2-3
    for i in range(4):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=2, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    # conv-block-4
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    shape_before_flattening = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Flatten()(x)
    z_mean = tf.keras.layers.Dense(latent_dim)(x)
    z_log_var = tf.keras.layers.Dense(latent_dim)(x)

    model = tf.keras.Model(inputs, (z_mean, z_log_var), name="Encoder")
    return model, shape_before_flattening