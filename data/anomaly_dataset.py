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

    def __new__(cls, input_dir_path, image_shape=(224, 224, 1)):
        cls.image_shape = image_shape
        cls.image_path_list = cls.__prepare_image_path_list(input_dir_path)

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
    def __prepare_image_path_list(self, input_dir_path):
        image_path_list = []
        for file in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
            image_path_list.extend(glob.glob(os.path.join(input_dir_path, '**', f'{file}'), recursive=True))
        return sorted(image_path_list)

    @classmethod
    def _generator(cls):
        while True:
            image_path = random.choice(cls.image_path_list)
            input_image, _, _ = ops.read_image(image_path, (cls.image_shape[0], cls.image_shape[1]))
            input_image = tf.convert_to_tensor(np.expand_dims(input_image, axis=-1).astype(np.uint8))
            yield input_image, input_image
