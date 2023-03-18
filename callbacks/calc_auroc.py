import glob
import os
from PIL import Image
from sklearn import metrics
import numpy as np

from data import ops


def prepare_inf_gt_images(raw_image_dir_path, gt_image_dir_path, target_shape):
    # read
    raw_image_path_list = []
    for file in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        raw_image_path_list.extend(glob.glob(os.path.join(raw_image_dir_path, '**', f'{file}'), recursive=True))
    raw_image_path_list = sorted(raw_image_path_list)

    raw_image_list = []
    gt_image_list = []
    category_list = []
    for raw_image_path in raw_image_path_list:
        category = raw_image_path.split('/')[-2]
        raw_image_list.append(ops.read_image(raw_image_path, target_shape)[0])
        category_list.append(category)
        if category != 'good':
            gt_image_path_list = glob.glob(os.path.join(gt_image_dir_path, category, os.path.basename(raw_image_path).replace('.', '_mask.')))
            gt_image_path = gt_image_path_list[0]
            gt_image_list.append(ops.read_image(gt_image_path, (target_shape[0], target_shape[1]))[0])
        else:
            gt_image_list.append(np.zeros((raw_image_list[-1].shape[0], raw_image_list[-1].shape[1], 1), dtype=raw_image_list[-1].dtype))
    return np.asarray(raw_image_list), np.asarray(gt_image_list), category_list

def instance_auroc_mean(inf_raw_image_list, gt_image_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    inf_raw_mean_list = [np.mean(inf_raw_image) for inf_raw_image in inf_raw_image_list]
    return instance_auroc(gt_labels, inf_raw_mean_list)


def instance_auroc_max(inf_raw_image_list, gt_image_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    inf_raw_max_list = [np.max(inf_raw_image) for inf_raw_image in inf_raw_image_list]
    return instance_auroc(gt_labels, inf_raw_max_list)


def instance_auroc(gt_labels, inf_raw_list):
    fpr, tpr, thresholds = metrics.roc_curve(
        gt_labels, inf_raw_list
    )
    auroc_metric = metrics.roc_auc_score(
        gt_labels, inf_raw_list
    )
    return auroc_metric, {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}


def full_pixel_auroc(inf_raw_image_list, gt_image_list):
    gt_image_array = (np.asarray(gt_image_list) > 125).flatten().astype(np.uint8)
    inf_raw_image_array = np.asarray(inf_raw_image_list).flatten()
    fpr, tpr, thresholds = metrics.roc_curve(
        gt_image_array, inf_raw_image_array
    )
    auroc_metric = metrics.roc_auc_score(
        gt_image_array, inf_raw_image_array
    )
    return auroc_metric, {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}


def anomaly_pixel_auroc(inf_raw_image_list, gt_image_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    gt_anomaly_indexes = np.asarray(gt_labels) == True
    gt_image_list = [gt_image for index, gt_image in enumerate(gt_image_list) if gt_anomaly_indexes[index]]
    inf_raw_image_list = [inf_raw_image for index, inf_raw_image in enumerate(inf_raw_image_list) if
                          gt_anomaly_indexes[index]]
    return full_pixel_auroc(inf_raw_image_list, gt_image_list)


def anomaly_detail_pixel_auroc(inf_raw_image_list, gt_image_list, inf_image_path_list):
    gt_labels = [np.max(gt_image) > 125 for gt_image in gt_image_list]
    gt_anomaly_indexes = np.asarray(gt_labels) == True
    gt_image_list = [gt_image for index, gt_image in enumerate(gt_image_list) if gt_anomaly_indexes[index]]
    inf_raw_image_list = [inf_raw_image for index, inf_raw_image in enumerate(inf_raw_image_list) if
                          gt_anomaly_indexes[index]]
    inf_image_path_list = [inf_image_path for index, inf_image_path in enumerate(inf_image_path_list) if
                           gt_anomaly_indexes[index]]

    auroc_metric_list, detail_list = [], []
    for inf_raw_image, gt_image, inf_image_path in zip(inf_raw_image_list, gt_image_list, inf_image_path_list):
        auroc_metric, detail_dict = full_pixel_auroc([inf_raw_image], [gt_image])
        auroc_metric_list.append((inf_image_path, auroc_metric))
        detail_list.append(detail_dict)
    return auroc_metric_list, detail_list
