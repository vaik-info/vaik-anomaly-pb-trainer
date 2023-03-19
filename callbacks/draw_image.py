from PIL import Image
import numpy as np
import os


def draw_image(y_true, y_pred, y_inf, output_dir_path, auroc_valid_category_list=None, rescale=255.):
    os.makedirs(output_dir_path, exist_ok=True)
    inf_raw_mean_list = [np.mean(inf_raw_image) for inf_raw_image in y_inf]
    for index in range(y_true.shape[0]):
        ground_truth_image, generated_image, generated_branch_image = y_true[index], y_pred[index], y_inf[index]
        ground_truth_image = Image.fromarray(ground_truth_image)
        if auroc_valid_category_list is None:
            ground_truth_image.save(os.path.join(output_dir_path, f'{inf_raw_mean_list[index]:.4f}_{index:04d}_gt.png'))
        else:
            ground_truth_image.save(os.path.join(output_dir_path, f'{inf_raw_mean_list[index]:.4f}_{index:04d}_{auroc_valid_category_list[index]}_gt.png'))

        generated_image = np.clip(generated_image*rescale, 0., 255.).astype(np.uint8)
        generated_image = Image.fromarray(generated_image)
        if auroc_valid_category_list is None:
            generated_image.save(os.path.join(output_dir_path, f'{inf_raw_mean_list[index]:.4f}_{index:04d}_pred.png'))
        else:
            generated_image.save(os.path.join(output_dir_path, f'{inf_raw_mean_list[index]:.4f}_{index:04d}_{auroc_valid_category_list[index]}_pred.png'))

        generated_branch_image = np.clip(generated_branch_image*rescale, 0., 255.).astype(np.uint8)

        generated_branch_image = Image.fromarray(generated_branch_image)
        if auroc_valid_category_list is None:
            generated_branch_image.save(os.path.join(output_dir_path, f'{inf_raw_mean_list[index]:.4f}_{index:04d}_branch.png'))
        else:
            generated_branch_image.save(os.path.join(output_dir_path, f'{inf_raw_mean_list[index]:.4f}_{index:04d}_{auroc_valid_category_list[index]}_branch.png'))


