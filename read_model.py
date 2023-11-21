# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 18:52:26 2023

@author: boyan
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


#check CUDA and GPUs

print(f'version:{torch.__version__}')
print(f'cuda_version:{torch.version.cuda}')
print(f'cuda_device_number:{torch.cuda.device_count()}')
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
            
        return image, label


# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets
train_dataset = CRC_Dataset(root_dir='E:\\BIC Project\\NCT-CRC-HE-100K', transform=transform)
val_dataset = CRC_Dataset(root_dir='E:\\BIC Project\\CRC-VAL-HE-7K', transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Initialize the model for multi-GPU if available
model = EfficientNet.from_pretrained('efficientnet-b0')
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, len(train_dataset.class_to_idx))

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
if num_gpus > 1:
    model = nn.DataParallel(model)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

output_dir = 'E:\\BIC Project\\20231109_CRC_EfficientNetb0_lr1e-4_batch128'
for epoch in range(15):
    checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch+1}.pth')
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_total = 0
    cancer_set = {7, 8}
    with torch.no_grad():
        for inputs, labels in val_loader:
            for i in range(len(labels)):
                if labels[i] in cancer_set:
                    break
            labels = labels[i:]
            inputs = inputs[i:]
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            val_total += inputs.size(0)
    
    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_corrects / val_total
    print(f'Epoch: {epoch+1} Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')