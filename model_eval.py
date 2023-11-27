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
model = nn.DataParallel(model)
model.to(device)

# Scan for checkpoint files
checkpoint_files = glob.glob(os.path.join(args.model_input_dir, '*.pth'))

# Prepare CSV file to log results
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(args.output_dir, f'model_eval_{current_time}.csv')
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['checkpoint_filename', 'val_accuracy', 'cancer_accuracy'])
# Initialize variables to track the highest combined accuracy and the corresponding checkpoint
highest_combined_acc = 0
highest_cancer_accuracy = 0
highest_val_accuracy = 0
best_checkpoint = ''

# Evaluate each model checkpoint
for checkpoint_path in tqdm(checkpoint_files, desc="Processing Checkpoints"):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Reset metrics
    val_corrects = 0
    tum_corrects = 0
    str_corrects = 0
    cancer_total = sum([1 for _, label in val_dataset if label == val_dataset.class_to_idx['STR']])
    cancer_total += sum([1 for _, label in val_dataset if label == val_dataset.class_to_idx['TUM']])

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

    # Calculate accuracies
    val_accuracy = val_corrects / len(val_dataset)
    cancer_accuracy = (str_corrects + tum_corrects) / cancer_total if cancer_total > 0 else 0
    combined_acc = 0.5 * val_accuracy + 0.5 * cancer_accuracy  # Weighted average of the two accuracies

    # Write to CSV
    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([os.path.basename(checkpoint_path), val_accuracy, cancer_accuracy])

    # Check and update best model based on combined accuracy
    if combined_acc > highest_combined_acc:
        highest_combined_acc = combined_acc
        highest_cancer_accuracy = cancer_accuracy
        highest_val_accuracy = val_accuracy
        best_checkpoint = checkpoint_path



# Load the best model for final evaluation
model.load_state_dict(torch.load(best_checkpoint, map_location=device))
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
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=val_dataset.class_to_idx.keys(), output_dict=True)

fpr_tum, tpr_tum, thresholds_tum = roc_curve(y_true, y_pred, pos_label=val_dataset.class_to_idx['TUM'])
roc_auc_tum = auc(fpr_tum, tpr_tum)
fpr_str, tpr_str, thresholds_str = roc_curve(y_true, y_pred, pos_label=val_dataset.class_to_idx['STR'])
roc_auc_str = auc(fpr_str, tpr_str)

tum_index = val_dataset.class_to_idx['TUM']
tn_tum = np.sum(cm) - np.sum(cm[tum_index, :]) - np.sum(cm[:, tum_index]) + cm[tum_index, tum_index]
fp_tum = np.sum(cm[:, tum_index]) - cm[tum_index, tum_index]
fn_tum = np.sum(cm[tum_index, :]) - cm[tum_index, tum_index]
tp_tum = cm[tum_index, tum_index]

tum_specificity = tn_tum / (tn_tum + fp_tum) if (tn_tum + fp_tum) > 0 else 0
tum_sensitivity = tp_tum / (tp_tum + fn_tum) if (tp_tum + fn_tum) > 0 else 0

str_index = val_dataset.class_to_idx['STR']
tn_str = np.sum(cm) - np.sum(cm[str_index, :]) - np.sum(cm[:, str_index]) + cm[str_index, str_index]
fp_str = np.sum(cm[:, str_index]) - cm[str_index, str_index]
fn_str = np.sum(cm[str_index, :]) - cm[str_index, str_index]
tp_str = cm[str_index, str_index]

str_specificity = tn_str / (tn_str + fp_str) if (tn_str + fp_str) > 0 else 0
str_sensitivity = tp_str / (tp_str + fn_str) if (tp_str + fn_str) > 0 else 0

# Confusion matrix for best model
class_names = list(val_dataset.class_to_idx.keys())
plot_confusion_matrix(cm, class_names)
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_confusion_matrix.png'))
#plt.show()

# Print and save results
results_str = f"""
Best model based on combined accuracy is: {best_checkpoint}
combined_accuracy: {highest_combined_acc}
val_accuracy:{highest_val_accuracy}
tum_accuracy:{highest_cancer_accuracy}

Confusion Matrix:
{cm}`

Classification Report:
{classification_report(y_true, y_pred, target_names=val_dataset.class_to_idx.keys())}

TUM Class Specificity: {tum_specificity}
TUM Class Sensitivity: {tum_sensitivity}
STR Class Specificity: {str_specificity}
STR Class Sensitivity: {str_sensitivity}
ROC AUC TUM: {roc_auc_tum}
ROC AUC STR: {roc_auc_str}
"""

print(results_str)

# Saving results to a file
results_filename = os.path.join(args.output_dir, f'{output_dir_name}_evaluation_results_{current_time}.txt')
with open(results_filename, 'w') as file:
    file.write(results_str)

# ROC Curve for TUM class

plt.figure()
plt.plot(fpr_tum, tpr_tum, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_tum:.2f})')
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
plt.plot(fpr_str, tpr_str, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_str:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for STR Class')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_roc_curve_str.png'))
#plt.show()