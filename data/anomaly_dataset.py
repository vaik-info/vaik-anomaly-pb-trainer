import glob
import os
import random
import tensorflow as tf
import numpy as np

from data import ops


class AnomalyDataset:
    image_path_list = None
    image_shape = None
    output_signature = None
    transform = None
    noize_transform=None

    def __new__(cls, input_dir_path, image_shape=(224, 224, 1), transform=None, noize_transform=None):
        cls.image_shape = image_shape
        cls.image_path_list = cls.__prepare_image_path_list(input_dir_path, image_shape)
        cls.transform = transform
        cls.noize_transform = noize_transform
        cls.output_signature = (
            tf.TensorSpec(name=f'input_image', shape=image_shape, dtype=tf.uint8),
            tf.TensorSpec(name=f'output_image', shape=image_shape, dtype=tf.uint8)
        )
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=cls.output_signature
        )
        return dataset

    @classmethod
    def __prepare_image_path_list(cls, input_dir_path, image_shape):
        image_path_list = []
        for file in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_path_list.extend(glob.glob(os.path.join(input_dir_path, '**', f'{file}'), recursive=True))
        image_list = []
        for image_path in image_path_list:
            input_image, _, _ = ops.read_image(image_path, image_shape)
            image_list.append(input_image)
        return image_list

    @classmethod
    def _generator(cls):
        while True:
            image = random.choice(cls.image_path_list)
            if cls.transform is None:
                image = tf.convert_to_tensor(image.astype(np.uint8))
            else:
                image = cls.transform(image=image)['image']

            if cls.noize_transform is None:
                noize_image = image
            else:
                noize_image = cls.noize_transform(image=image)['image']
            yield tf.convert_to_tensor(noize_image), tf.convert_to_tensor(image)
