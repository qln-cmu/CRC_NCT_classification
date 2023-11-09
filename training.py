#!/usr/bin/env python

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
train_dataset = CRC_Dataset(root_dir='C:\\Jason\\CRC_NRT_data\\NCT-CRC-HE-100K', transform=transform)
val_dataset = CRC_Dataset(root_dir='C:\\Jason\\CRC_NRT_data\\CRC-VAL-HE-7K', transform=transform)

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

# Output directory for checkpoints and logs
output_dir = 'C:\\Jason\\CRC_NRT_data\\20231109_CRC_EfficientNetb0_lr1e-4_batch128'
os.makedirs(output_dir, exist_ok=True)

# CSV log file setup
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = os.path.join(output_dir, f'training_log_{current_time}.csv')
with open(log_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['epoch', 'train_acc', 'train_loss', 'val_acc', 'val_loss', 'current_lr'])

# Training loop with tqdm
num_epochs = 15
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
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print stats and save to CSV
    print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} - '
          f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} - LR: {current_lr}')
    with open(log_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, epoch_acc.item(), epoch_loss, val_epoch_acc.item(), val_epoch_loss, current_lr])
    
    # Save the model checkpoint
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch+1}_{current_time}.pth')
    torch.save(model.state_dict(), checkpoint_path)

print("Training complete!")
