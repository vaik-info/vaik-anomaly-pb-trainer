from sklearn import metrics
import numpy as np

def instance_auroc_mean(good_mse_array, anomaly_mse_array):
    gt_labels = [False, ] * good_mse_array.shape[0]
    gt_labels.extend([True, ] * anomaly_mse_array.shape[0])
    mse_array = np.concatenate([good_mse_array, anomaly_mse_array], axis=0)
    mse_mean_list = [np.mean(mse) for mse in mse_array]
    return instance_auroc(gt_labels, mse_mean_list)


def instance_auroc_max(good_mse_array, anomaly_mse_array):
    gt_labels = [False, ] * good_mse_array.shape[0]
    gt_labels.extend([True, ] * anomaly_mse_array.shape[0])
    mse_array = np.concatenate([good_mse_array, anomaly_mse_array], axis=0)
    mse_mean_list = [np.max(mse) for mse in mse_array]
    return instance_auroc(gt_labels, mse_mean_list)


def instance_auroc(gt_labels, inf_raw_list):
    fpr, tpr, thresholds = metrics.roc_curve(
        gt_labels, inf_raw_list
    )
    auroc_metric = metrics.roc_auc_score(
        gt_labels, inf_raw_list
    )
    return auroc_metric, {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}