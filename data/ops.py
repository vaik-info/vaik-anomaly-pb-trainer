from typing import Tuple
from PIL import Image, ImageOps
import numpy as np


def read_image(image_path: str, target_shape: Tuple[int, int] = (256, 256, 3)):
    if len(target_shape) == 3 and target_shape[-1] == 3:
        pil_image = Image.open(image_path).convert('RGB')
    else:
        pil_image = Image.open(image_path).convert('L')
    pil_image, padding_bottom_right, org_image_shape = __resize_and_pad(pil_image, target_shape)
    np_image = np.asarray(pil_image)
    np_image = np.expand_dims(np_image, axis=-1) if len(np_image.shape) == 2 else np_image
    return np_image


def __resize_and_pad(pil_image, target_shape):
    width, height = pil_image.size

    scale = min(target_shape[0] / height, target_shape[1] / width)

    resize_width = int(width * scale)
    resize_height = int(height * scale)

    pil_image = pil_image.resize((resize_width, resize_height))
    padding_bottom, padding_right = target_shape[0] - resize_height, target_shape[1] - resize_width
    pil_image = ImageOps.expand(pil_image, (0, 0, padding_right, padding_bottom), fill=0)
    return pil_image, (padding_bottom, padding_right), (height, width)
