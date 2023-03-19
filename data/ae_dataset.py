import glob
import os
import random
import tensorflow as tf

from data import ops


class AEDataset:
    image_path_list = None
    image_shape = None
    output_signature = None

    def __new__(cls, input_dir_path, image_shape=(224, 224, 3)):
        cls.image_shape = image_shape
        cls.image_path_list = cls.__prepare_image_path_list(input_dir_path, image_shape)
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
            image_list.append(image_path)
        return image_list

    @classmethod
    def _generator(cls):
        while True:
            input_image = ops.read_image(random.choice(cls.image_path_list), cls.image_shape)
            input_image = tf.convert_to_tensor(input_image)
            yield input_image, input_image
