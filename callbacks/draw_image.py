from PIL import Image
import numpy as np
import os


def draw_image(good_inf_images, good_mses, anomaly_inf_images, anomaly_mses, output_dir_path, rescale=255.):
    good_mse_mean_list = [np.mean(good_mse) for good_mse in good_mses]
    output_sub_dir_path = os.path.join(output_dir_path, 'good')
    os.makedirs(output_sub_dir_path, exist_ok=True)
    for good_index, good_inf_image in enumerate(good_inf_images):
        good_inf_image = np.clip(good_inf_image * rescale, 0, 255).astype(np.uint8)
        good_inf_image = Image.fromarray(good_inf_image)
        good_inf_image.save(os.path.join(output_sub_dir_path, f'{good_mse_mean_list[good_index]:.4f}.png'))

    anomaly_mse_mean_list = [np.mean(anomaly_mse) for anomaly_mse in anomaly_mses]
    output_sub_dir_path = os.path.join(output_dir_path, 'anomaly')
    os.makedirs(output_sub_dir_path, exist_ok=True)
    for anomaly_index, anomaly_inf_image in enumerate(anomaly_inf_images):
        anomaly_inf_image = np.clip(anomaly_inf_image * rescale, 0, 255).astype(np.uint8)
        anomaly_inf_image = Image.fromarray(anomaly_inf_image)
        anomaly_inf_image.save(os.path.join(output_sub_dir_path, f'{anomaly_mse_mean_list[anomaly_index]:.4f}.png'))

