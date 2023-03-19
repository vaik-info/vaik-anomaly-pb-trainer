import os
import argparse

import numpy as np
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

from data import ae_dataset
from models import memory_block

from callbacks import calc_auroc


def inference(encoder_model, data, memory_block_model, batch_size):
    features = encoder_model.predict(data, batch_size=batch_size)
    anomaly_scores = memory_block_model.predict(features)
    return anomaly_scores


def train(mse_model_dir_path, encoder_model_dir_path, train_image_dir_path, test_good_image_dir_path,
          test_anomaly_image_dir_path, sample_ratio, max_distance_ratio, random_ratio, batch_size, sample_num,
          output_file_path):
    mse_model = tf.keras.models.load_model(mse_model_dir_path)
    encoder_model = tf.keras.models.load_model(encoder_model_dir_path)

    ## train
    TrainDataset = type(f'TrainDataset', (ae_dataset.AEDataset,), dict())
    train_dataset = TrainDataset(train_image_dir_path, encoder_model.input_shape[1:])
    train_good_dataset = iter(train_dataset.padded_batch(batch_size=int(sample_num * sample_ratio), padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8))))
    features = next(iter(train_dataset.padded_batch(batch_size=int(sample_num * random_ratio), padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))[0].numpy()

    ## valid good
    TestGoodDataset = type(f'TestGoodDataset', (ae_dataset.AEDataset,), dict())
    valid_good_dataset = TestGoodDataset(test_good_image_dir_path, encoder_model.input_shape[1:])
    valid_good_data = next(iter(valid_good_dataset.padded_batch(batch_size=100, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))

    ## valid anomaly
    TestAnomalyDataset = type(f'TestAnomalyDataset', (ae_dataset.AEDataset,), dict())
    valid_anomaly_dataset = TestAnomalyDataset(test_anomaly_image_dir_path, encoder_model.input_shape[1:])
    valid_anomaly_data = next(iter(valid_anomaly_dataset.padded_batch(batch_size=100, padding_values=(
        tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.uint8)))))


    for train_good_data in train_good_dataset:
        mse_array = mse_model.predict(train_good_data[0], batch_size)
        mse_array = np.asarray([np.mean(mse) for mse in mse_array])
        mse_max_indexes = np.argsort(-mse_array)
        mse_max_train_good_data = train_good_data[0].numpy()[mse_max_indexes[:int(len(mse_max_indexes)*max_distance_ratio)]]
        features = np.concatenate([features, mse_max_train_good_data], axis=0)
        if features.shape[0] > sample_num:
            break
        else:
            print(f'extract feature: {features.shape[0]}/{sample_num}')


    ## encode
    features = encoder_model.predict(features, batch_size=batch_size)
    memory_block_model = memory_block.FaissNearestNeighbour()

    # Train Memory Block
    memory_block_model.train(features)

    # Save Memory Block
    memory_block_model.save(output_file_path)

    good_scores = inference(encoder_model, valid_good_data[0], memory_block_model, batch_size)
    anomaly_scores = inference(encoder_model, valid_anomaly_data[0], memory_block_model, batch_size)

    auroc = calc_auroc.instance_auroc_mean(good_scores, anomaly_scores)
    print(auroc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train faiss')
    parser.add_argument('--mse_model_dir_path', type=str,
                        default='/home/kentaro/.vaik_anomaly_pb_trainer/output_model/2023-03-19-21-55-35/epoch-0016_steps-1000_batch-16_loss-26.2637_val_loss-25.9032_val_anomaly_loss-54.6594_mse-0.0138-val_mse-0.0145_val_anomaly_mse-0.0384_val_auroc_mean-0.9600/mse_model')
    parser.add_argument('--encoder_model_dir_path', type=str,
                        default='/home/kentaro/.vaik_anomaly_pb_trainer/output_model/2023-03-19-21-55-35/epoch-0016_steps-1000_batch-16_loss-26.2637_val_loss-25.9032_val_anomaly_loss-54.6594_mse-0.0138-val_mse-0.0145_val_anomaly_mse-0.0384_val_auroc_mean-0.9600/encoder_sampler_model')
    parser.add_argument('--train_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/train/good')
    parser.add_argument('--test_good_image_dir_path', type=str, default='~/.vaik-mnist-anomaly-dataset/valid/good')
    parser.add_argument('--test_anomaly_image_dir_path', type=str,
                        default='~/.vaik-mnist-anomaly-dataset/valid/anomaly')
    parser.add_argument('--total_sample_num', type=int, default=4000)
    parser.add_argument('--sample_ratio', type=int, default=0.25)
    parser.add_argument('--max_distance_ratio', type=int, default=0.05)
    parser.add_argument('--random_ratio', type=int, default=0.1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file_path', type=str, default='~/.vaik_anomaly_pb_trainer/output_faiss/output.faiss')
    args = parser.parse_args()

    args.mse_model_dir_path = os.path.expanduser(args.mse_model_dir_path)
    args.encoder_model_dir_path = os.path.expanduser(args.encoder_model_dir_path)
    args.train_image_dir_path = os.path.expanduser(args.train_image_dir_path)
    args.test_good_image_dir_path = os.path.expanduser(args.test_good_image_dir_path)
    args.test_anomaly_image_dir_path = os.path.expanduser(args.test_anomaly_image_dir_path)
    args.output_file_path = os.path.expanduser(args.output_file_path)

    os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)

    train(args.mse_model_dir_path, args.encoder_model_dir_path, args.train_image_dir_path,
          args.test_good_image_dir_path,
          args.test_anomaly_image_dir_path, args.sample_ratio, args.max_distance_ratio, args.random_ratio,
          args.batch_size, args.total_sample_num, args.output_file_path)
