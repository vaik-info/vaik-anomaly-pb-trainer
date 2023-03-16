import os
import argparse
import time
from datetime import datetime
import pytz
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from data import anomaly_dataset
from models import cvae_encoder, cvae_sampler, cvae_decoder

from losses import vae_loss


def train(train_image_dir_path, test_image_dir_path, epoch_size, step_size, batch_size, image_height, image_width,
          latent_dim,
          test_max_sample, output_dir_path):
    input_shape = (image_height, image_width, 1)
    # Prepare Dataset
    ## train
    TrainDataset = type(f'TrainDataset', (anomaly_dataset.AnomalyDataset,), dict())
    train_dataset = TrainDataset(train_image_dir_path, input_shape)
    train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))

    ## valid
    TestDataset = type(f'TestDataset', (anomaly_dataset.AnomalyDataset,), dict())
    valid_dataset = TestDataset(test_image_dir_path, input_shape)
    valid_dataset = next(iter(valid_dataset.padded_batch(batch_size=test_max_sample, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))

    # Prepare Model
    ## encoder
    encoder_model, shape_before_flattening = cvae_encoder.prepare(input_shape, latent_dim=latent_dim)
    ## sampler
    sampler_model = cvae_sampler.prepare(mean_input=encoder_model.outputs[0], log_var_input=encoder_model.outputs[1])
    ## decoder
    decoder_model = cvae_decoder.decoder(sampler_model.outputs[0], shape_before_flattening, input_shape)
    ## optimizer
    encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # train
    @tf.function
    def train_step(input_images, output_images):
        with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
            mean, log_var = encoder_model(input_images, training=True)
            latent = sampler_model([mean, log_var])
            generated_images = decoder_model(latent, training=True)
            loss = vae_loss.vae_loss(output_images, generated_images, mean, log_var)

        gradients_of_enc = encoder.gradient(loss, encoder_model.trainable_variables)
        gradients_of_dec = decoder.gradient(loss, decoder_model.trainable_variables)

        encoder_optimizer.apply_gradients(zip(gradients_of_enc, encoder_model.trainable_variables))
        decoder_optimizer.apply_gradients(zip(gradients_of_dec, decoder_model.trainable_variables))
        return tf.reduce_sum(loss)

    for image_batch in iter(train_dataset):
        loss = train_step(image_batch[0], image_batch[1])
        print(f'{loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/train/good')
    parser.add_argument('--test_image_dir_path', type=str,
                        default='~/.vaik-mnist-anomaly-dataset/valid/anomaly/test/good')
    parser.add_argument('--epoch_size', type=int, default=100)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--latent_dim', type=int, default=2)
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik_anomaly_pb_trainer/output')
    args = parser.parse_args()

    args.train_image_dir_path = os.path.expanduser(args.train_image_dir_path)
    args.test_image_dir_path = os.path.expanduser(args.test_image_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(os.path.dirname(args.output_dir_path), exist_ok=True)

    train(args.train_image_dir_path, args.test_image_dir_path, args.epoch_size, args.step_size, args.batch_size,
          args.image_height, args.image_width, args.latent_dim, args.test_max_sample, args.output_dir_path)
