import os
import argparse
from datetime import datetime
import pytz
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import tqdm
import cv2

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
import albumentations as A
from models import dae
from losses import vae_loss
from callbacks import draw_image, calc_auroc


def train(train_image_dir_path, test_image_dir_path,
          auroc_valid_raw_image_dir_path, auroc_valid_gt_image_dir_path,
          epoch_size, step_size, batch_size, image_height, image_width,
          latent_dim,
          test_max_sample, output_dir_path):
    input_shape = (image_height, image_width, 3)

    # Prepare Train Dataset
    transform = A.Compose([
        A.ShiftScaleRotate(always_apply=True)
    ])

    noize_transform = A.Compose([
        A.GaussNoise(),
        A.ISONoise(),
        A.Cutout(num_holes=16, max_h_size=32, max_w_size=32)
    ])
    ## train
    TrainDataset = type(f'TrainDataset', (anomaly_dataset.AnomalyDataset,), dict())
    train_dataset = TrainDataset(train_image_dir_path, input_shape, transform, noize_transform)
    train_dataset = iter(train_dataset.padded_batch(batch_size=batch_size, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8))))

    ## valid
    TestDataset = type(f'TestDataset', (anomaly_dataset.AnomalyDataset,), dict())
    valid_dataset = TestDataset(test_image_dir_path, input_shape)
    valid_dataset = next(iter(valid_dataset.padded_batch(batch_size=test_max_sample, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))

    # Prepare Auroc Dataset
    ## valid
    auroc_valid_raw_image_array, auroc_valid_gt_image_array, auroc_valid_category_list = calc_auroc.prepare_inf_gt_images(
        auroc_valid_raw_image_dir_path, auroc_valid_gt_image_dir_path, input_shape)

    # Prepare Model
    ## encoder
    dae_model = dae.prepare(input_shape)
    dae_model.summary()

    ## optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')

    # train
    @tf.function
    def train_step(input_images, output_images):
        with tf.GradientTape() as encoder, tf.GradientTape() as decoder:
            generated_images = dae_model(input_images, training=True)

            loss =tfa.losses.sigmoid_focal_crossentropy(tf.cast(output_images, tf.float32) / 255., generated_images)
            mse = tf.keras.metrics.mse(tf.cast(output_images, tf.float32) / 255., generated_images)

        gradients_of_enc = encoder.gradient(loss, dae_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients_of_enc, dae_model.trainable_variables))
        return tf.reduce_mean(loss), tf.reduce_mean(mse), generated_images

    # test
    @tf.function
    def test_step(input_images, output_images):
        generated_images = dae_model(input_images, training=True)

        loss =tfa.losses.sigmoid_focal_crossentropy(tf.cast(output_images, tf.float32) / 255., generated_images)
        mse = tf.keras.metrics.mse(tf.cast(output_images, tf.float32) / 255., generated_images)
        return tf.reduce_mean(loss), tf.reduce_mean(mse), generated_images

    # test
    @tf.function
    def test_auroc_step(input_images):
        generated_images = dae_model(input_images, training=True)
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
            val_loss, val_mse, val_generated_images = test_step(valid_dataset[0], valid_dataset[1])
            print(f'val_loss:{float(val_loss):.4f}, val_mse:{float(val_mse):.4f}')

            # calc auroc
            auroc_valid_inf_image_array = test_auroc_step(auroc_valid_raw_image_array)
            valid_instance_auroc, _ = calc_auroc.instance_auroc_mean(auroc_valid_inf_image_array, auroc_valid_gt_image_array)
            valid_mean_instance_auroc, _ = calc_auroc.instance_auroc_mean(auroc_valid_inf_image_array, auroc_valid_gt_image_array)
            valid_max_instance_auroc, _ = calc_auroc.instance_auroc_max(auroc_valid_inf_image_array, auroc_valid_gt_image_array)
            valid_full_pixel_instance_auroc, _ = calc_auroc.full_pixel_auroc(auroc_valid_inf_image_array, auroc_valid_gt_image_array)
            valid_anomaly_pixel_instance_auroc, _ = calc_auroc.anomaly_pixel_auroc(auroc_valid_inf_image_array, auroc_valid_gt_image_array)

            print(f'valid_instance_auroc(mean): {valid_mean_instance_auroc:.4f}')
            print(f'valid_instance_auroc(max): {valid_max_instance_auroc:.4f}')
            print(f'valid_full_pixel_auroc: {valid_full_pixel_instance_auroc:.4f}')
            print(f'valid_anomaly_pixel_auroc: {valid_anomaly_pixel_instance_auroc:.4f}')

            # draw image
            save_model_sub_dir_path = os.path.join(save_model_dir_path,
                                                   f'epoch-{epoch:04d}_steps-{step_size}_batch-{batch_size}_loss-{float(np.mean(train_loss_list)):.4f}_val_loss-{val_loss:.4f}_mses-{float(np.mean(train_mse_list)):.4f}_val_mse-{val_mse:.4f}_valid_full_pixel_auroc-{valid_full_pixel_instance_auroc:.4f}')
            save_model_train_sub_dir_path = os.path.join(save_model_sub_dir_path, 'train_generated_images')
            draw_image.draw_image(image_batch[1].numpy(), train_generated_images.numpy(), save_model_train_sub_dir_path)

            save_model_val_sub_dir_path = os.path.join(save_model_sub_dir_path, 'val_generated_images')
            draw_image.draw_image(valid_dataset[1].numpy(), val_generated_images.numpy(), save_model_val_sub_dir_path)


            _, _, auroc_valid_inf_image_array = test_step(auroc_valid_raw_image_array, auroc_valid_raw_image_array)
            save_model_val_anomaly_sub_dir_path = os.path.join(save_model_sub_dir_path, 'val_anomaly_generated_images')
            draw_image.draw_image(auroc_valid_raw_image_array, auroc_valid_inf_image_array.numpy(), save_model_val_anomaly_sub_dir_path, auroc_valid_category_list)

            # save model
            dae_model.save(os.path.join(save_model_sub_dir_path, 'dae_model'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/train/raw/good')
    parser.add_argument('--test_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid/raw/good')
    parser.add_argument('--auroc_valid_raw_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid/raw')
    parser.add_argument('--auroc_valid_gt_image_dir_path', type=str,
                        default='~/.vaik-mnist-anomaly-dataset/valid/ground_truth')
    parser.add_argument('--epoch_size', type=int, default=1000)
    parser.add_argument('--step_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--test_max_sample', type=int, default=100)
    parser.add_argument('--output_dir_path', type=str, default='~/.vaik_anomaly_pb_trainer/output')
    args = parser.parse_args()

    args.train_image_dir_path = os.path.expanduser(args.train_image_dir_path)
    args.auroc_valid_raw_image_dir_path = os.path.expanduser(args.auroc_valid_raw_image_dir_path)
    args.auroc_valid_gt_image_dir_path = os.path.expanduser(args.auroc_valid_gt_image_dir_path)
    args.test_image_dir_path = os.path.expanduser(args.test_image_dir_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(os.path.dirname(args.output_dir_path), exist_ok=True)

    train(args.train_image_dir_path, args.test_image_dir_path,
          args.auroc_valid_raw_image_dir_path, args.auroc_valid_gt_image_dir_path,
          args.epoch_size, args.step_size, args.batch_size,
          args.image_height, args.image_width, args.latent_dim, args.test_max_sample, args.output_dir_path)