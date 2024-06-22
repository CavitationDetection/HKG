import os
import torch
import numpy as np
import pickle
import os
import torch
import csv
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,roc_curve, auc


# create some dirs
def create_dirs(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)


def calculate_mAP(y_true, y_pred):
    """
    Calculate mean Average Precision (mAP) for a multi-label classification task.
    
    Args:
    y_true: Tensor, true labels (ground truth), shape (num_samples, num_classes)
    y_pred: Tensor, predicted probabilities, shape (num_samples, num_classes)
    
    Returns:
    mAP: float, mean Average Precision
    """
    aps = []
    
    for i in range(y_true.shape[1]):
        y_true_class = y_true[:, i]
        y_pred_class = y_pred[:, i]
        ap = average_precision_score(y_true_class, y_pred_class)
        aps.append(ap)
    
    mAP = np.mean(aps)
    
    return mAP

def write_metrics_to_csv(metrics, metric_names):
    """
    Writes metrics to CSV files, each metric in a separate file with the file name being the metric name.

    Args:
        metrics (list): A list containing metric values, each metric is a list of floating-point numbers.
        metric_names (list): A list containing metric names, each name is a string.
    """
    for i, metric in enumerate(metrics):
        metric_name = metric_names[i]
        csv_file = f'./output_results/{metric_name}.csv'

        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)

            # If the file is empty, write the header row
            if file.tell() == 0:
                header = ['Data', metric_name]
                writer.writerow(header)

            # Write the values of the metric
            row = [len(metric), *metric]
            writer.writerow(row)

        print(f"Metrics '{metric_name}' have been appended to {csv_file}")


def choose_threshold_based_on_auc(y_true, y_probabilities):
    thresholds = []
    
    for i in range(y_true.shape[1]):  
        fpr, tpr, threshold = roc_curve(y_true[:, i].cpu().detach().numpy(), y_probabilities[:, i].cpu().detach().numpy())
        roc_auc = auc(fpr, tpr)
        best_threshold_index = np.argmax(tpr - fpr)  
        best_threshold = threshold[best_threshold_index]
        thresholds.append(best_threshold)
    return thresholds

def log_metrics_to_file(metrics, filename):
    # open log file and write
    log_file = open(filename, 'a')

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write("Timestamp: {}\n".format(current_time))

    for metric_name, metric_value in metrics.items():
        log_file.write("{}: {}\n".format(metric_name, metric_value))
    log_file.write("-" * 40 + "\n")
    log_file.close()




