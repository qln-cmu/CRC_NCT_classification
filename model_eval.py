#!/usr/bin/env python
"""
Model Evaluation Script for CRC Dataset

This script, named 'model_eval.py', is designed for the evaluation of trained models on the CRC (Colorectal Cancer)
dataset using the EfficientNet architecture. It systematically assesses multiple model checkpoints to identify
the most effective one based on combined accuracy metrics.

Key Functionalities:
- Loads validation data from a specified directory, with support for resizing images.
- Initializes the EfficientNet model specified by the user.
- Evaluates each model checkpoint found in the given input directory.
- Calculates both overall validation accuracy and class-specific accuracy for the 'TUM' class.
- Selects the best model based on a combined metric: 0.5 * validation accuracy + 0.5 * 'TUM' class accuracy.
- Generates a confusion matrix and a classification report, including specificity, sensitivity, and ROC AUC for the 'TUM' class.
- Outputs evaluation results in both visual (plots) and textual (csv and txt files) formats.

Usage:
Run this script with the required command line arguments:
- --val_data_dir: Path to the directory containing validation data.
- --model_input_dir: Directory containing the saved model checkpoints.
- --model_name: Name of the EfficientNet model used during training.
- --output_dir: Directory to save the evaluation results.
- --resized_dim: The dimensions for resizing the input images (default 224).

Example Command:
python model_eval.py --val_data_dir "path/to/validation/data>" \
                     --model_input_dir "path/to/model/checkpoints" \
                     --model_name "efficientnet-b0" \
                     --output_dir "path/to/output/directory" \
                     --resized_dim 224
"""

import os
import csv
import torch
import torch.nn as nn
import argparse
import glob
import itertools
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Evaluate a trained model on CRC dataset.')
parser.add_argument('--val_data_dir', type=str, help='Path to validation data directory.')
parser.add_argument('--model_input_dir', type=str, help='Input directory to model checkpoints and train logs.')
parser.add_argument('--model_name', type=str, help='Model architecture used during training.')
parser.add_argument('--output_dir', type=str, help='Output directory to save all results.')
parser.add_argument('--resized_dim', default=224, type=int, help='Resize dimension for input images.')
args = parser.parse_args()
output_dir_name = os.path.basename(args.output_dir)

# Define a dataset class
class CRC_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}

        for label in os.listdir(root_dir):
            for img_file in os.listdir(os.path.join(root_dir, label)):
                self.image_paths.append(os.path.join(root_dir, label, img_file))
                self.labels.append(self.class_to_idx[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Function to calculate class-wise accuracy
def class_wise_accuracy(conf_matrix):
    accuracies = []
    for i in range(len(conf_matrix)):
        if conf_matrix[i].sum() == 0:
            accuracies.append(0)
        else:
            accuracies.append(conf_matrix[i, i] / conf_matrix[i].sum())
    return accuracies

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title=f'{output_dir_name} Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Define transforms
transform = transforms.Compose([
    transforms.Resize((args.resized_dim, args.resized_dim)),  # Resize the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load validation dataset
val_dataset = CRC_Dataset(root_dir=args.val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False) #can increase batch_size for faster validation

os.makedirs(args.output_dir, exist_ok=True)

# Initialize the model
model = EfficientNet.from_name(args.model_name)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(val_dataset.class_to_idx))

# Define the device and check for multiple GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
if args.resized_dim != 112 or args.model_name != 'efficientnet-b1':
    model = nn.DataParallel(model)
model.to(device)

# Scan for checkpoint files
checkpoint_files = glob.glob(os.path.join(args.model_input_dir, '*.pth'))

# Prepare CSV file to log results
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(args.output_dir, f'model_eval_{current_time}.csv')
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['checkpoint_filename', "val_accuracy", "str_recall", "str_prec", "str_f1", "str_f1w", "tum_recall", "tum_prec", "tum_f1", "tum_f1w"])
# Initialize variables to track the highest combined accuracy and the corresponding checkpoint
# highest_combined_acc = 0
highest_str_f1 = 0
highest_tum_f1 = 0
highest_str_f1w = 0
highest_tum_f1w = 0
highest_val_accuracy = 0
best_checkpoint_val = ''
best_checkpoint_str = ''
best_checkpoint_tum = ''
best_checkpoint_strw = ''
best_checkpoint_tumw= ''

# Evaluate each model checkpoint
for checkpoint_path in tqdm(checkpoint_files, desc="Processing Checkpoints"):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Reset metrics
    val_corrects = 0
    tum_corrects = 0
    str_corrects = 0
    tum_all = 0
    str_all = 0
    str_total = sum([1 for _, label in val_dataset if label == val_dataset.class_to_idx['STR']])
    tum_total = sum([1 for _, label in val_dataset if label == val_dataset.class_to_idx['TUM']])

    # Validation loop
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            val_corrects += torch.sum(preds == labels.data).item()
            str_corrects += torch.sum((preds == labels.data) & (labels == val_dataset.class_to_idx['STR'])).item()
            tum_corrects += torch.sum((preds == labels.data) & (labels == val_dataset.class_to_idx['TUM'])).item()
            str_all += torch.sum(preds == val_dataset.class_to_idx['STR']).item()
            tum_all += torch.sum(preds == val_dataset.class_to_idx['TUM']).item()
            

    # Calculate accuracies
    val_accuracy = val_corrects / len(val_dataset)
    str_recall = str_corrects / str_total if str_total > 0 else 0
    tum_recall = tum_corrects / tum_total if tum_total > 0 else 0
    str_prec = str_corrects / str_all if str_all > 0 else 0
    tum_prec = tum_corrects / tum_all if tum_all > 0 else 0
    str_f1 = 2 * str_prec*str_recall / (str_prec + str_recall)
    tum_f1 = 2 * tum_prec*tum_recall / (tum_prec + tum_recall)
    
    alpha = 1.2
    str_f1w = (1 + alpha**2) * str_prec*str_recall / (alpha**2*str_prec + str_recall)
    tum_f1w = (1 + alpha**2) * tum_prec*tum_recall / (alpha**2*tum_prec + tum_recall)
    
    
    

    # Write to CSV
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([os.path.basename(checkpoint_path), val_accuracy, str_recall, str_prec, str_f1, str_f1w, tum_recall, tum_prec, tum_f1, tum_f1w])

    # Check and update best model based on combined accuracy
    if val_accuracy > highest_val_accuracy:
        highest_val_accuracy = val_accuracy
        best_checkpoint_val = checkpoint_path
    if str_f1 > highest_str_f1:
        highest_str_f1 = str_f1
        best_checkpoint_str = checkpoint_path
    if tum_f1 > highest_tum_f1:
        highest_tum_f1 = tum_f1
        best_checkpoint_tum = checkpoint_path
    if str_f1w > highest_str_f1w:
        highest_str_f1w = str_f1w
        best_checkpoint_strw = checkpoint_path
    if tum_f1w > highest_tum_f1w:
        highest_tum_f1w = tum_f1w
        best_checkpoint_tumw = checkpoint_path


# Overall best model
# Load the best model for final evaluation
model.load_state_dict(torch.load(best_checkpoint_val, map_location=device))
model.eval()

# Reinitialize y_true and y_pred for evaluation of best model
y_true, y_pred = [], []

# Evaluation loop for best model
for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics calculation
cm_val = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=val_dataset.class_to_idx.keys(), output_dict=True)

fpr_tum, tpr_tum, thresholds_tum = roc_curve(y_true, y_pred, pos_label=val_dataset.class_to_idx['TUM'])
roc_auc_tum = auc(fpr_tum, tpr_tum)
fpr_str, tpr_str, thresholds_str = roc_curve(y_true, y_pred, pos_label=val_dataset.class_to_idx['STR'])
roc_auc_str = auc(fpr_str, tpr_str)

tum_index = val_dataset.class_to_idx['TUM']
tn_tum = np.sum(cm_val) - np.sum(cm_val[tum_index, :]) - np.sum(cm_val[:, tum_index]) + cm_val[tum_index, tum_index]
fp_tum = np.sum(cm_val[:, tum_index]) - cm_val[tum_index, tum_index]
fn_tum = np.sum(cm_val[tum_index, :]) - cm_val[tum_index, tum_index]
tp_tum = cm_val[tum_index, tum_index]

tum_specificity = tn_tum / (tn_tum + fp_tum) if (tn_tum + fp_tum) > 0 else 0
tum_sensitivity = tp_tum / (tp_tum + fn_tum) if (tp_tum + fn_tum) > 0 else 0

str_index = val_dataset.class_to_idx['STR']
tn_str = np.sum(cm_val) - np.sum(cm_val[str_index, :]) - np.sum(cm_val[:, str_index]) + cm_val[str_index, str_index]
fp_str = np.sum(cm_val[:, str_index]) - cm_val[str_index, str_index]
fn_str = np.sum(cm_val[str_index, :]) - cm_val[str_index, str_index]
tp_str = cm_val[str_index, str_index]

str_specificity = tn_str / (tn_str + fp_str) if (tn_str + fp_str) > 0 else 0
str_sensitivity = tp_str / (tp_str + fn_str) if (tp_str + fn_str) > 0 else 0

# Confusion matrix for best model
class_names = list(val_dataset.class_to_idx.keys())
plt.figure()
plot_confusion_matrix(cm_val, class_names)
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_val_confusion_matrix.png'))
#plt.show()
plt.close()

# Best STR model
# Load the best model for final evaluation
model.load_state_dict(torch.load(best_checkpoint_str, map_location=device))
model.eval()

# Reinitialize y_true and y_pred for evaluation of best model
y_true_str, y_pred_str = [], []

# Evaluation loop for best model
for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true_str.extend(labels.cpu().numpy())
        y_pred_str.extend(preds.cpu().numpy())

# Metrics calculation
cm_str = confusion_matrix(y_true_str, y_pred_str)
report_str = classification_report(y_true_str, y_pred_str, target_names=val_dataset.class_to_idx.keys(), output_dict=True)

fpr_tum_str, tpr_tum_str, thresholds_tum_str = roc_curve(y_true_str, y_pred_str, pos_label=val_dataset.class_to_idx['TUM'])
roc_auc_tum_str = auc(fpr_tum_str, tpr_tum_str)
fpr_str_str, tpr_str_str, thresholds_str = roc_curve(y_true_str, y_pred_str, pos_label=val_dataset.class_to_idx['STR'])
roc_auc_str_str = auc(fpr_str_str, tpr_str_str)

tum_index = val_dataset.class_to_idx['TUM']
tn_tum = np.sum(cm_str) - np.sum(cm_str[tum_index, :]) - np.sum(cm_str[:, tum_index]) + cm_str[tum_index, tum_index]
fp_tum = np.sum(cm_str[:, tum_index]) - cm_str[tum_index, tum_index]
fn_tum = np.sum(cm_str[tum_index, :]) - cm_str[tum_index, tum_index]
tp_tum = cm_str[tum_index, tum_index]

tum_specificity_str = tn_tum / (tn_tum + fp_tum) if (tn_tum + fp_tum) > 0 else 0
tum_sensitivity_str = tp_tum / (tp_tum + fn_tum) if (tp_tum + fn_tum) > 0 else 0

str_index = val_dataset.class_to_idx['STR']
tn_str = np.sum(cm_str) - np.sum(cm_str[str_index, :]) - np.sum(cm_str[:, str_index]) + cm_str[str_index, str_index]
fp_str = np.sum(cm_str[:, str_index]) - cm_str[str_index, str_index]
fn_str = np.sum(cm_str[str_index, :]) - cm_str[str_index, str_index]
tp_str = cm_str[str_index, str_index]

str_specificity_str = tn_str / (tn_str + fp_str) if (tn_str + fp_str) > 0 else 0
str_sensitivity_str = tp_str / (tp_str + fn_str) if (tp_str + fn_str) > 0 else 0

# Confusion matrix for best model
class_names = list(val_dataset.class_to_idx.keys())
plt.figure()
plot_confusion_matrix(cm_str, class_names)
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_str_confusion_matrix.png'))
#plt.show()
plt.close()

# Best TUM model
# Load the best model for final evaluation
model.load_state_dict(torch.load(best_checkpoint_tum, map_location=device))
model.eval()

# Reinitialize y_true and y_pred for evaluation of best model
y_true_tum, y_pred_tum = [], []

# Evaluation loop for best model
for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true_tum.extend(labels.cpu().numpy())
        y_pred_tum.extend(preds.cpu().numpy())

# Metrics calculation
cm_tum = confusion_matrix(y_true_tum, y_pred_tum)
report_tum = classification_report(y_true_tum, y_pred_tum, target_names=val_dataset.class_to_idx.keys(), output_dict=True)

fpr_tum_tum, tpr_tum_tum, thresholds_tum_tum = roc_curve(y_true_tum, y_pred_tum, pos_label=val_dataset.class_to_idx['TUM'])
roc_auc_tum_tum = auc(fpr_tum_tum, tpr_tum_tum)
fpr_str_tum, tpr_str_tum, thresholds_str_tum = roc_curve(y_true_tum, y_pred_tum, pos_label=val_dataset.class_to_idx['STR'])
roc_auc_str_tum = auc(fpr_str_tum, tpr_str_tum)

tum_index = val_dataset.class_to_idx['TUM']
tn_tum = np.sum(cm_tum) - np.sum(cm_tum[tum_index, :]) - np.sum(cm_tum[:, tum_index]) + cm_tum[tum_index, tum_index]
fp_tum = np.sum(cm_tum[:, tum_index]) - cm_tum[tum_index, tum_index]
fn_tum = np.sum(cm_tum[tum_index, :]) - cm_tum[tum_index, tum_index]
tp_tum = cm_tum[tum_index, tum_index]

tum_specificity_tum = tn_tum / (tn_tum + fp_tum) if (tn_tum + fp_tum) > 0 else 0
tum_sensitivity_tum = tp_tum / (tp_tum + fn_tum) if (tp_tum + fn_tum) > 0 else 0

str_index = val_dataset.class_to_idx['STR']
tn_str = np.sum(cm_tum) - np.sum(cm_tum[str_index, :]) - np.sum(cm_tum[:, str_index]) + cm_tum[str_index, str_index]
fp_str = np.sum(cm_tum[:, str_index]) - cm_tum[str_index, str_index]
fn_str = np.sum(cm_tum[str_index, :]) - cm_tum[str_index, str_index]
tp_str = cm_tum[str_index, str_index]

str_specificity_tum = tn_str / (tn_str + fp_str) if (tn_str + fp_str) > 0 else 0
str_sensitivity_tum = tp_str / (tp_str + fn_str) if (tp_str + fn_str) > 0 else 0

# Confusion matrix for best model
class_names = list(val_dataset.class_to_idx.keys())
plt.figure()
plot_confusion_matrix(cm_tum, class_names)
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_tum_confusion_matrix.png'))
#plt.show()
plt.close()


# Best Weighted STR model
# Load the best model for final evaluation
model.load_state_dict(torch.load(best_checkpoint_strw, map_location=device))
model.eval()

# Reinitialize y_true and y_pred for evaluation of best model
y_true_strw, y_pred_strw = [], []

# Evaluation loop for best model
for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true_strw.extend(labels.cpu().numpy())
        y_pred_strw.extend(preds.cpu().numpy())

# Metrics calculation
cm_strw = confusion_matrix(y_true_strw, y_pred_strw)
report_strw = classification_report(y_true_strw, y_pred_strw, target_names=val_dataset.class_to_idx.keys(), output_dict=True)

fpr_tum_strw, tpr_tum_strw, thresholds_tum_strw = roc_curve(y_true_strw, y_pred_strw, pos_label=val_dataset.class_to_idx['TUM'])
roc_auc_tum_strw = auc(fpr_tum_strw, tpr_tum_strw)
fpr_str_strw, tpr_str_strw, thresholds_strw = roc_curve(y_true_strw, y_pred_strw, pos_label=val_dataset.class_to_idx['STR'])
roc_auc_str_strw = auc(fpr_str_strw, tpr_str_strw)

tum_index = val_dataset.class_to_idx['TUM']
tn_tumw = np.sum(cm_strw) - np.sum(cm_strw[tum_index, :]) - np.sum(cm_strw[:, tum_index]) + cm_strw[tum_index, tum_index]
fp_tumw = np.sum(cm_strw[:, tum_index]) - cm_strw[tum_index, tum_index]
fn_tumw = np.sum(cm_strw[tum_index, :]) - cm_strw[tum_index, tum_index]
tp_tumw = cm_strw[tum_index, tum_index]

tum_specificity_strw = tn_tumw / (tn_tumw + fp_tumw) if (tn_tumw + fp_tumw) > 0 else 0
tum_sensitivity_strw = tp_tumw / (tp_tumw + fn_tumw) if (tp_tumw + fn_tumw) > 0 else 0

str_index = val_dataset.class_to_idx['STR']
tn_strw = np.sum(cm_strw) - np.sum(cm_strw[str_index, :]) - np.sum(cm_strw[:, str_index]) + cm_strw[str_index, str_index]
fp_strw = np.sum(cm_strw[:, str_index]) - cm_strw[str_index, str_index]
fn_strw = np.sum(cm_strw[str_index, :]) - cm_strw[str_index, str_index]
tp_strw = cm_strw[str_index, str_index]

str_specificity_strw = tn_strw / (tn_strw + fp_strw) if (tn_strw + fp_strw) > 0 else 0
str_sensitivity_strw = tp_strw / (tp_strw + fn_strw) if (tp_strw + fn_strw) > 0 else 0

# Confusion matrix for best model
class_names = list(val_dataset.class_to_idx.keys())
plt.figure()
plot_confusion_matrix(cm_strw, class_names)
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_weighted_str_confusion_matrix.png'))
#plt.show()
plt.close()

# Best Weighted TUM model
# Load the best model for final evaluation
model.load_state_dict(torch.load(best_checkpoint_tumw, map_location=device))
model.eval()

# Reinitialize y_true and y_pred for evaluation of best model
y_true_tumw, y_pred_tumw = [], []

# Evaluation loop for best model
for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_true_tumw.extend(labels.cpu().numpy())
        y_pred_tumw.extend(preds.cpu().numpy())

# Metrics calculation
cm_tumw = confusion_matrix(y_true_tumw, y_pred_tumw)
report_tumw = classification_report(y_true_tumw, y_pred_tumw, target_names=val_dataset.class_to_idx.keys(), output_dict=True)

fpr_tum_tumw, tpr_tum_tumw, thresholds_tum_tumw = roc_curve(y_true_tumw, y_pred_tumw, pos_label=val_dataset.class_to_idx['TUM'])
roc_auc_tum_tumw = auc(fpr_tum_tumw, tpr_tum_tumw)
fpr_str_tumw, tpr_str_tumw, thresholds_str_tumw = roc_curve(y_true_tumw, y_pred_tumw, pos_label=val_dataset.class_to_idx['STR'])
roc_auc_str_tumw = auc(fpr_str_tumw, tpr_str_tumw)

tum_index = val_dataset.class_to_idx['TUM']
tn_tumw = np.sum(cm_tumw) - np.sum(cm_tumw[tum_index, :]) - np.sum(cm_tumw[:, tum_index]) + cm_tumw[tum_index, tum_index]
fp_tumw = np.sum(cm_tumw[:, tum_index]) - cm_tumw[tum_index, tum_index]
fn_tumw = np.sum(cm_tumw[tum_index, :]) - cm_tumw[tum_index, tum_index]
tp_tumw = cm_tumw[tum_index, tum_index]

tum_specificity_tumw = tn_tumw / (tn_tumw + fp_tumw) if (tn_tumw + fp_tumw) > 0 else 0
tum_sensitivity_tumw = tp_tumw / (tp_tumw + fn_tumw) if (tp_tumw + fn_tumw) > 0 else 0

str_index = val_dataset.class_to_idx['STR']
tn_strw = np.sum(cm_tumw) - np.sum(cm_tumw[str_index, :]) - np.sum(cm_tumw[:, str_index]) + cm_tumw[str_index, str_index]
fp_strw = np.sum(cm_tumw[:, str_index]) - cm_tumw[str_index, str_index]
fn_strw = np.sum(cm_tumw[str_index, :]) - cm_tumw[str_index, str_index]
tp_strw = cm_tumw[str_index, str_index]

str_specificity_tumw = tn_strw / (tn_strw + fp_strw) if (tn_strw + fp_strw) > 0 else 0
str_sensitivity_tumw = tp_strw / (tp_strw + fn_strw) if (tp_strw + fn_strw) > 0 else 0

# Confusion matrix for best model
class_names = list(val_dataset.class_to_idx.keys())
plt.figure()
plot_confusion_matrix(cm_tumw, class_names)
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_weighted_tum_confusion_matrix.png'))
#plt.show()
plt.close()

# Print and save results
results_str = f"""
Best model based on validation accuracy is: {best_checkpoint_val}
val_accuracy:{highest_val_accuracy}

Confusion Matrix:
{cm_val}`

Classification Report:
{classification_report(y_true, y_pred, target_names=val_dataset.class_to_idx.keys())}

TUM Class Specificity: {tum_specificity}
TUM Class Sensitivity: {tum_sensitivity}
STR Class Specificity: {str_specificity}
STR Class Sensitivity: {str_sensitivity}
ROC AUC TUM: {roc_auc_tum}
ROC AUC STR: {roc_auc_str}

Best model based on STR F1-score is: {best_checkpoint_str}
str_f1:{highest_str_f1}

Confusion Matrix:
{cm_str}`

Classification Report:
{classification_report(y_true_str, y_pred_str, target_names=val_dataset.class_to_idx.keys())}

TUM Class Specificity: {tum_specificity_str}
TUM Class Sensitivity: {tum_sensitivity_str}
STR Class Specificity: {str_specificity_str}
STR Class Sensitivity: {str_sensitivity_str}
ROC AUC TUM: {roc_auc_tum_str}
ROC AUC STR: {roc_auc_str_str}


Best model based on TUM F1-score is: {best_checkpoint_tum}
tum_f1:{highest_tum_f1}

Confusion Matrix:
{cm_tum}`

Classification Report:
{classification_report(y_true_tum, y_pred_tum, target_names=val_dataset.class_to_idx.keys())}

TUM Class Specificity: {tum_specificity_tum}
TUM Class Sensitivity: {tum_sensitivity_tum}
STR Class Specificity: {str_specificity_tum}
STR Class Sensitivity: {str_sensitivity_tum}
ROC AUC TUM: {roc_auc_tum_tum}
ROC AUC STR: {roc_auc_str_tum}

Best model based on STR Weighted F1-score is: {best_checkpoint_strw}
str_f1:{highest_str_f1w}

Confusion Matrix:
{cm_strw}`

Classification Report:
{classification_report(y_true_strw, y_pred_strw, target_names=val_dataset.class_to_idx.keys())}

TUM Class Specificity: {tum_specificity_strw}
TUM Class Sensitivity: {tum_sensitivity_strw}
STR Class Specificity: {str_specificity_strw}
STR Class Sensitivity: {str_sensitivity_strw}
ROC AUC TUM: {roc_auc_tum_strw}
ROC AUC STR: {roc_auc_str_strw}


Best model based on TUM Weighted F1-score is: {best_checkpoint_tumw}
tum_f1:{highest_tum_f1w}

Confusion Matrix:
{cm_tumw}`

Classification Report:
{classification_report(y_true_tumw, y_pred_tumw, target_names=val_dataset.class_to_idx.keys())}

TUM Class Specificity: {tum_specificity_tumw}
TUM Class Sensitivity: {tum_sensitivity_tumw}
STR Class Specificity: {str_specificity_tumw}
STR Class Sensitivity: {str_sensitivity_tumw}
ROC AUC TUM: {roc_auc_tum_tumw}
ROC AUC STR: {roc_auc_str_tumw}
"""

print(results_str)

# Saving results to a file
results_filename = os.path.join(args.output_dir, f'{output_dir_name}_evaluation_results_{current_time}.txt')
with open(results_filename, 'w') as file:
    file.write(results_str)

# ROC Curve for TUM class

plt.figure()
plt.plot(fpr_tum, tpr_tum, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_tum_tum:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for TUM Class')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_roc_curve_tum.png'))
#plt.show()

# ROC Curve for STR class

plt.figure()
plt.plot(fpr_str, tpr_str, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_str_str:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for STR Class')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_roc_curve_str.png'))
#plt.show()

# ROC Curve for Weighted TUM class

plt.figure()
plt.plot(fpr_tum, tpr_tum, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_tum_tumw:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Weighted TUM Class')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_roc_curve_tum_weighted.png'))
#plt.show()

# ROC Curve for Weighted STR class

plt.figure()
plt.plot(fpr_str, tpr_str, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_str_strw:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Weighted STR Class')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_roc_curve_str_weighted.png'))
#plt.show()

