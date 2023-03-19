import os
import argparse
from datetime import datetime
import pytz
import tensorflow as tf
import numpy as np
import tqdm

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

from data import ae_dataset
from models import cvae_encoder, cvae_sampler, cvae_decoder
from losses import vae_loss
from callbacks import draw_image, calc_auroc


def train(train_image_dir_path, test_good_image_dir_path, test_anomaly_image_dir_path,
          epoch_size, step_size, batch_size, image_height, image_width, latent_dim, test_max_sample, output_dir_path):
    input_shape = (image_height, image_width, 3)

    ## train
    TrainDataset = type(f'TrainDataset', (ae_dataset.AEDataset,), dict())
    train_dataset = TrainDataset(train_image_dir_path, input_shape)
    train_good_data = next(iter(train_dataset.padded_batch(batch_size=test_max_sample, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))
    train_dataset = iter(train_dataset.padded_batch(batch_size=batch_size, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8))))

    ## valid good
    TestGoodDataset = type(f'TestGoodDataset', (ae_dataset.AEDataset,), dict())
    valid_good_dataset = TestGoodDataset(test_good_image_dir_path, input_shape)
    valid_good_data = next(iter(valid_good_dataset.padded_batch(batch_size=test_max_sample, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))

    ## valid anomaly
    TestAnomalyDataset = type(f'TestAnomalyDataset', (ae_dataset.AEDataset,), dict())
    valid_anomaly_dataset = TestAnomalyDataset(test_anomaly_image_dir_path, input_shape)
    valid_anomaly_data = next(iter(valid_anomaly_dataset.padded_batch(batch_size=test_max_sample, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))

    # Prepare Model
    ## encoder
    encoder_model, shape_before_flattening, (z_mean, z_log_var) = cvae_encoder.prepare(input_shape,
                                                                                       latent_dim=latent_dim)
    ## sampler
    sampler_model, z = cvae_sampler.prepare(mean_input=z_mean, log_var_input=z_log_var)
    ## decoder
    decoder_model, outputs = cvae_decoder.decoder(z, shape_before_flattening, input_shape)
    ## all_model
    all_model = tf.keras.Model(encoder_model.inputs, outputs)
    all_model.summary()

    ## optimizer
    encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')

    # train
    @tf.function
    def train_step(input_images, output_images):
        with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
            mean, log_var = encoder_model(input_images, training=True)
            latent = sampler_model([mean, log_var])
            generated_images = decoder_model(latent, training=True)

            loss = vae_loss.vae_loss(output_images, generated_images, mean, log_var)
            mse = tf.keras.metrics.mse(tf.cast(output_images, tf.float32) / 255., generated_images)

        gradients_of_enc = encoder.gradient(loss, encoder_model.trainable_variables)
        gradients_of_dec = decoder.gradient(loss, decoder_model.trainable_variables)

        encoder_optimizer.apply_gradients(zip(gradients_of_enc, encoder_model.trainable_variables))
        decoder_optimizer.apply_gradients(zip(gradients_of_dec, decoder_model.trainable_variables))
        return tf.reduce_mean(loss), tf.reduce_mean(mse), generated_images

    # test
    @tf.function
    def test_step(input_images, output_images):
        mean, log_var = encoder_model(input_images, training=True)
        latent = sampler_model([mean, log_var])
        generated_images = decoder_model(latent, training=True)

        loss = vae_loss.vae_loss(output_images, generated_images, mean, log_var)
        mse_array = tf.math.squared_difference(generated_images, tf.cast(output_images, tf.float32) / 255.)
        return tf.reduce_mean(loss), tf.reduce_mean(mse_array), generated_images, mse_array

    # test
    @tf.function
    def test_auroc_step(input_images):
        mean, log_var = encoder_model(input_images, training=True)
        latent = sampler_model([mean, log_var])
        generated_images = decoder_model(latent, training=True)
        mse = tf.keras.metrics.mse(tf.cast(input_images, tf.float32) / 255., generated_images)
        return mse

    for epoch in range(epoch_size):
        with tqdm.tqdm(range(step_size), unit="steps") as monitor_tqdm:
            train_loss_list = []
            train_mse_list = []
            for step in monitor_tqdm:
                # train
                monitor_tqdm.set_description(f"Epoch {epoch}")
                image_batch = next(train_dataset)
                train_loss, train_mse, train_generated_images = train_step(image_batch[0], image_batch[1])
                train_loss_list.append(train_loss)
                train_mse_list.append(train_mse)
                monitor_tqdm.set_postfix(loss=float(np.mean(train_loss_list)), mse=float(np.mean(train_mse_list)))

            # valid
            train_loss, train_mse, train_generated_images, train_mse_array = test_step(train_good_data[0],
                                                                                       train_good_data[1])
            val_loss, val_mse, val_generated_images, val_mse_array = test_step(valid_good_data[0], valid_good_data[1])
            val_anomaly_loss, val_anomaly_mse, val_anomaly_generated_images, val_anomaly_mse_array = test_step(
                valid_anomaly_data[0],
                valid_anomaly_data[1])

            val_auroc_mean, val_auroc_mean_detail = calc_auroc.instance_auroc_mean(val_mse_array, val_anomaly_mse_array)
            val_auroc_max, val_auroc_max_detail = calc_auroc.instance_auroc_max(val_mse_array, val_anomaly_mse_array)

            print(f'loss:{float(train_loss):.4f}, val_loss:{float(val_loss):.4f}, val_anomaly_loss:{float(val_anomaly_loss):.4f}, '
                f'mse:{float(train_mse):.4f}, val_mse:{float(val_mse):.4f}, val_anomaly_mse:{float(val_anomaly_mse):.4f}',
                f'val_auroc_mean:{float(val_auroc_mean):.4f}, val_auroc_max:{float(val_auroc_max):.4f}')

            # save model
            save_model_sub_dir_path = os.path.join(save_model_dir_path,
                                                   f'epoch-{epoch:04d}_steps-{step_size}_batch-{batch_size}_'
                                                   f'loss-{float(train_loss):.4f}_val_loss-{float(val_loss):.4f}_val_anomaly_loss-{float(val_anomaly_loss):.4f}_'
                                                   f'mse-{float(train_mse):.4f}-val_mse-{float(val_mse):.4f}_val_anomaly_mse-{float(val_anomaly_mse):.4f}_'
                                                   f'val_auroc_mean-{float(val_auroc_mean)}_val_auroc_max-{float(val_auroc_max)}')
            os.makedirs(save_model_sub_dir_path, exist_ok=True)
            all_model.save(os.path.join(save_model_sub_dir_path, 'all_model'))

            output_draw_dir_path = os.path.join(save_model_sub_dir_path, 'draw_image')
            draw_image.draw_image(valid_good_data[0].numpy(), val_generated_images, val_mse_array, valid_anomaly_data[0].numpy(), val_anomaly_generated_images, val_anomaly_mse_array, output_draw_dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/train/good')
    parser.add_argument('--test_good_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid/good')
    parser.add_argument('--test_anomaly_image_dir_path', type=str,
                        default='~/.vaik-mnist-anomaly-dataset/valid/anomaly')
    parser.add_argument('--epoch_size', type=int, default=1000)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--latent_dim', type=int, default=3)
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik_anomaly_pb_trainer/output')
    args = parser.parse_args()

    args.train_image_dir_path = os.path.expanduser(args.train_image_dir_path)
    args.test_good_image_dir_path = os.path.expanduser(args.test_good_image_dir_path)
    args.test_anomaly_image_dir_path = os.path.expanduser(args.test_anomaly_image_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(os.path.dirname(args.output_dir_path), exist_ok=True)

    train(args.train_image_dir_path, args.test_good_image_dir_path, args.test_anomaly_image_dir_path,
          args.epoch_size, args.step_size, args.batch_size,
          args.image_height, args.image_width, args.latent_dim, args.test_max_sample, args.output_dir_path)
