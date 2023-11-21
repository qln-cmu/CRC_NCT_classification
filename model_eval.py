#!/usr/bin/env python

import os
import csv
import torch
import torch.nn as nn
import argparse
import glob
import datetime
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
args = parser.parse_args()


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


# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load validation dataset
val_dataset = CRC_Dataset(root_dir=args.val_data_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
output_dir_name = os.path.basename(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)

# Initialize the model
model = EfficientNet.from_name(args.model_name)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(val_dataset.class_to_idx))

# Find the best model based on validation accuracy
log_path = os.path.join(args.model_input_dir, max(os.listdir(args.model_input_dir), key=lambda x: 'train_log' in x))
best_epoch = 0
best_val_acc = 0

with open(log_path, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        val_acc = float(row['val_acc'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = int(row['epoch'])

# Assuming checkpoint file name format: '..._epoch_{best_epoch}_...'
checkpoint_pattern = os.path.join(args.model_input_dir, f"*_epoch_{best_epoch}_*.pth")
checkpoint_files = glob.glob(checkpoint_pattern)

if not checkpoint_files:
    raise FileNotFoundError(f"No checkpoint file found for epoch {best_epoch} in directory {args.model_input_dir}")

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Evaluate the model
y_true, y_pred = [], []
checkpoint_path = checkpoint_files[0]  # Assuming there's only one match per epoch
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

for inputs, labels in val_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    y_true.extend(labels.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

# Metrics calculation
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=val_dataset.class_to_idx.keys(), output_dict=True)
specificity = cm[0,0] / (cm[0,0] + cm[0,1])
sensitivity = cm[1,1] / (cm[1,0] + cm[1,1])
fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=val_dataset.class_to_idx['TUM'])
roc_auc = auc(fpr, tpr)

# Print and save results
results_str = f"""
Confusion Matrix:
{cm}

Classification Report:
{classification_report(y_true, y_pred, target_names=val_dataset.class_to_idx.keys())}

Specificity: {specificity}
Sensitivity: {sensitivity}
ROC AUC: {roc_auc}
"""

print(results_str)

# Saving results to a file
results_filename = os.path.join(args.output_dir, f'{output_dir_name}_evaluation_results_{current_time}.txt')
with open(results_filename, 'w') as file:
    file.write(results_str)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.output_dir, f'{output_dir_name}_roc_curve_{current_time}.png'))
plt.show()