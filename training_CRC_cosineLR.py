#!/usr/bin/env python

"""
Train EfficientNet on CRC Dataset

This script trains an EfficientNet model on the CRC (Colorectal Cancer) dataset. It allows for training on both single
and multi-GPU setups. The script includes options for data augmentation through image resizing and normalization.
The trained model is saved along with training logs that include accuracy and loss metrics for both training and
validation datasets. The script accepts command-line arguments for various parameters like data directory paths,
model type, learning rate, and output directory for saved models and logs.

Arguments:
- train_data_dir: Directory containing training data.
- val_data_dir: Directory containing validation data.
- model_name: Type of EfficientNet model to be used (default: efficientnet-b0).
- output_dir: Directory where model checkpoints and logs will be saved.
- resized_dim: Size to which input images will be resized (default: 224).
- learning_rate: Learning rate for the optimizer (default: 1e-3).

The script checks for available GPUs and uses CUDA for training if available. Multi-GPU training is supported and automatically detected.

Example usage:
python training_CRC_cosineLR.py --train_data_dir "path_to_training_data" \
                                --val_data_dir "path_to_validation_data" \
                                --model_name "efficientnet-b0" \
                                --output_dir "path_to_output_directory" \
                                --resized_dim 224 \
                                --learning_rate 1e-3
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from tqdm import tqdm
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a model on CRC dataset.')
parser.add_argument('--train_data_dir', default='C:\\Jason\\CRC_NRT_data\\NCT-CRC-HE-100K', type=str, help='Path to training data directory.')
parser.add_argument('--val_data_dir', default='C:\\Jason\\CRC_NRT_data\\CRC-VAL-HE-7K', type=str, help='Path to validation data directory.')
parser.add_argument('--model_name', default='efficientnet-b0', type=str, help='Model name for EfficientNet.')
parser.add_argument('--output_dir', default='C:\\Jason\\CRC_NRT_data\\CRC_EfficientNetb3_lr1e-3_batch128_cosineLR_noImageNet_Norm',
                    type=str, help='Output directory for checkpoints and logs.')
parser.add_argument('--resized_dim', default=224, type=int, help='Resize dimension for input images.')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate for training.')
args = parser.parse_args()

# Check CUDA and GPUs
print(f'version:{torch.__version__}')
print(f'cuda_version:{torch.version.cuda}')
print(f'cuda_device_number:{torch.cuda.device_count()}')
if torch.cuda.device_count() > 0:
    print(f'cuda_device_name:{torch.cuda.get_device_name(0)}')

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

        # Print the shape of the image tensor if you want to confirm
        #print(f'Shape of image tensor (after resizing): {image.size()}')

        return image, label


# Define transforms
transform = transforms.Compose([
    transforms.Resize((args.resized_dim, args.resized_dim)),  # Resize the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = CRC_Dataset(root_dir=args.train_data_dir, transform=transform)
val_dataset = CRC_Dataset(root_dir=args.val_data_dir, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize the model for multi-GPU if available
model = EfficientNet.from_pretrained(args.model_name)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(train_dataset.class_to_idx))

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
model = nn.DataParallel(model)
model.to(device)

# Training parameters
learning_rate = args.learning_rate
num_epochs = 15

# Define loss function, optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Output directory for checkpoints and logs
output_dir = args.output_dir
output_dir_name = os.path.basename(output_dir)
os.makedirs(output_dir, exist_ok=True)

# CSV log file setup
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"{output_dir_name}_train_log_{current_time}.csv"
log_path = os.path.join(output_dir, log_filename)
with open(log_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss', 'current_lr'])

# Training loop with tqdm
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_corrects = 0
    # Set up loop to train with number of epochs
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = train_loss / len(train_dataset)
    epoch_acc = train_corrects.double() / len(train_dataset)
    # Get the current learning rate for logging
    current_lr = scheduler.get_last_lr()[0]

    # Model validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_epoch_loss = val_loss / len(val_dataset)
    val_epoch_acc = val_corrects.double() / len(val_dataset)

    # Print stats and save to CSV
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} - '
          f'Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.4f} - LR: {current_lr}')
    with open(log_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, epoch_acc.item(), epoch_loss, val_epoch_acc.item(), val_epoch_loss, current_lr])

    # Step the scheduler - updates the learning rate for the next epoch
    scheduler.step()

    # Save the model checkpoint
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_filename = f"{output_dir_name}_epoch_{epoch+1}_{current_time}.pth"
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    torch.save(model.state_dict(), checkpoint_path)

print("Training complete!")
