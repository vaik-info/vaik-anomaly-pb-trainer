from typing import Tuple
from PIL import Image, ImageOps

import os
import glob
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def read_image_dir(input_dir_path, target_shape: Tuple[int, int, int] = (256, 256, 1)):
    image_path_list = []
    for file in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        image_path_list.extend(glob.glob(os.path.join(input_dir_path, '**', f'{file}'), recursive=True))
    image_path_list = sorted(image_path_list)

    image_array = np.zeros((len(image_path_list),) + target_shape, dtype=np.uint8)
    padding_bottom_right_list = []
    org_image_shape_list = []
    for image_index, image_path in enumerate(image_path_list):
        image_array[image_index], padding_bottom_right, org_image_shape = read_image(image_path, target_shape)
        padding_bottom_right_list.append(padding_bottom_right)
        org_image_shape_list.append(org_image_shape)
    return image_array, padding_bottom_right_list, org_image_shape_list, image_path_list


def read_image(image_path: str, target_shape: Tuple[int, int] = (256, 256)):
    pil_image = Image.open(image_path).convert('L')
    pil_image, padding_bottom_right, org_image_shape = __resize_and_pad(pil_image, target_shape)
    np_image = np.asarray(pil_image)
    np_image = np.expand_dims(np_image, axis=-1) if len(np_image.shape)==2 else np_image
    return np_image, padding_bottom_right, org_image_shape


def decode(anomaly_scores, padding_bottom_right_list, model_input_size, feature_extractor_model_output_size):
    anomaly_scores = np.reshape(anomaly_scores, (anomaly_scores.shape[0],) + feature_extractor_model_output_size + (1,))
    anomaly_scores = tf.image.resize(anomaly_scores, model_input_size)
    anomaly_scores = tfa.image.gaussian_filter2d(anomaly_scores).numpy()
    anomaly_scores_raw_images = []
    for anomaly_score, padding_bottom_right in tqdm(
            zip(anomaly_scores, padding_bottom_right_list), desc='ops.decode()'):
        anomaly_scores_raw_image = anomaly_score[:anomaly_score.shape[0] - padding_bottom_right[0],
                                   :anomaly_score.shape[1] - padding_bottom_right[1], :]
        anomaly_scores_raw_images.append(anomaly_scores_raw_image)
    return anomaly_scores_raw_images


def convert_min_max_normalize_images(anomaly_scores_raw_images):
    anomaly_min_max_normalize_images = []
    min_raw_list = []
    max_raw_list = []
    for anomaly_scores_raw_image in anomaly_scores_raw_images:
        min_raw = np.min(anomaly_scores_raw_image)
        max_raw = np.max(anomaly_scores_raw_image)
        anomaly_min_max_normalize_images.append((anomaly_scores_raw_image - min_raw) / (max_raw - min_raw))
        min_raw_list.append(min_raw)
        max_raw_list.append(max_raw)
    return anomaly_min_max_normalize_images, min_raw_list, max_raw_list


def convert_min_max_revert_images(anomaly_min_max_normalize_images, min_raw_list, max_raw_list):
    anomaly_min_max_revert_images = []
    for anomaly_min_max_normalize_image, min_raw, max_raw in zip(anomaly_min_max_normalize_images, min_raw_list,
                                                                 max_raw_list):
        anomaly_min_max_revert_images.append((anomaly_min_max_normalize_image * (max_raw - min_raw)) + min_raw)
    return anomaly_min_max_revert_images


def convert_resize_images(anomaly_scores_images, target_size=(224, 224)):
    anomaly_resize_images = []
    for anomaly_scores_image in anomaly_scores_images:
        if len(anomaly_scores_image.shape) == 2:
            anomaly_resize_image = tf.squeeze(tf.image.resize(tf.expand_dims(anomaly_scores_image, -1), target_size),
                                              -1).numpy()
        else:
            anomaly_resize_image = tf.image.resize(tf.expand_dims(anomaly_scores_image, -1), target_size).numpy()
        anomaly_resize_images.append(anomaly_resize_image)
    return anomaly_resize_images


def convert_rgb_images(anomaly_min_max_normalize_images):
    anomaly_rgb_images = []
    for anomaly_min_max_normalize_image in anomaly_min_max_normalize_images:
        anomaly_rgb_image = np.zeros(anomaly_min_max_normalize_image.shape[:2] + (3,), dtype=np.uint8)
        anomaly_image = np.clip(anomaly_min_max_normalize_image * 255, 0, 255).astype('uint8')
        anomaly_rgb_image[:, :, 0] = np.squeeze(anomaly_image, axis=-1)
        anomaly_rgb_images.append(anomaly_rgb_image)
    return anomaly_rgb_images


def __resize_and_pad(pil_image, target_shape):
    width, height = pil_image.size

    scale = min(target_shape[0] / height, target_shape[1] / width)

    resize_width = int(width * scale)
    resize_height = int(height * scale)

    pil_image = pil_image.resize((resize_width, resize_height))
    padding_bottom, padding_right = target_shape[0] - resize_height, target_shape[1] - resize_width
    pil_image = ImageOps.expand(pil_image, (0, 0, padding_right, padding_bottom), fill=0)
    return pil_image, (padding_bottom, padding_right), (height, width)
