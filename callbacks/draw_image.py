from PIL import Image
import numpy as np
import os


def draw_image(y_true, y_pred, output_dir_path, rescale=255.):
    os.makedirs(output_dir_path, exist_ok=True)
    for index in range(y_true.shape[0]):
        ground_truth_image, generated_image = y_true[index], y_pred[index]
        ground_truth_image = Image.fromarray(np.squeeze(ground_truth_image, axis=-1))
        ground_truth_image.save(os.path.join(output_dir_path, f'{index:04d}_gt.png'))

        generated_image = np.clip(generated_image*rescale, 0., 255.).astype(np.uint8)
        generated_image = Image.fromarray(np.squeeze(generated_image, axis=-1))
        generated_image.save(os.path.join(output_dir_path, f'{index:04d}_pred.png'))
