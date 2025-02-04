# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:57:41 2024

@author: tanbi
"""

import numpy as np
#%%

def compute_confusion_matrix(predictions, ground_truth):
    TP = np.sum((predictions == 1) & (ground_truth == 1))
    TN = np.sum((predictions == 0) & (ground_truth == 0))
    FP = np.sum((predictions == 1) & (ground_truth == 0))
    FN = np.sum((predictions == 0) & (ground_truth == 1))
    return TP, TN, FP, FN
#%%
def compute_accuracy(predictions, ground_truth):
    TP, TN, FP, FN = compute_confusion_matrix(predictions, ground_truth)
    return (TP + TN) / len(ground_truth)
#%%
def compute_sensitivity(predictions, ground_truth):
    TP, _, _, FN = compute_confusion_matrix(predictions, ground_truth)
    return TP / (TP + FN) if (TP + FN) != 0 else 0
#%%
def compute_specificity(predictions, ground_truth):
    _, TN, FP, _ = compute_confusion_matrix(predictions, ground_truth)
    return TN / (TN + FP) if (TN + FP) != 0 else 0
#%%
def compute_mcc(predictions, ground_truth):
    TP, TN, FP, FN = compute_confusion_matrix(predictions, ground_truth)
    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return numerator / denominator if denominator != 0 else 0
#%%
# Example Usage
predictions = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1])
ground_truth = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0])

# Compute Metrics
TP, TN, FP, FN = compute_confusion_matrix(predictions, ground_truth)
accuracy = compute_accuracy(predictions, ground_truth)
sensitivity = compute_sensitivity(predictions, ground_truth)
specificity = compute_specificity(predictions, ground_truth)
mcc = compute_mcc(predictions, ground_truth)

# Print Results
print("Confusion Matrix:")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy}")
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"MCC: {mcc}")
