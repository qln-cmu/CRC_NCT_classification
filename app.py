# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 00:18:05 2023

@author: boyan
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import PIL
import numpy as np
import stainNorm_Macenko
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class ClassifyModel:
    def __init__(self):
        self.model = None
        self.class2tag = {'ADI': 0, 'BACK': 1, 'DEB': 2, 'LYM': 3, 'MUC': 4, 'MUS': 5, 'NORM': 6, 'STR': 7, 'TUM': 8}
        self.tag2class = {
            0: "adipose",
            1: "background",
            2: "debris",
            3: "lymphocytes",
            4: "mucus",
            5: "smooth muscle",
            6: "normal colon mucosa",
            7: "cancer-associated stroma",
            8: "colorectal adenocarcinoma epithelium"
        }
        self.transform = None
        self.checkpoint_val = None
        self.checkpoint_str = None
        self.checkpoint_tum = None
        self.val_acc = 0.9638
        self.val_prec = 0.94
        self.str_recall = 0.8242
        self.tum_recall = 0.9984
        self.str_prec = 0.82
        self.tum_prec = 0.84
        

    def load(self):
        image_size = 112
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint_val = torch.load("model_result_112_b1_epoch_11_20231126_225903.pth", map_location=device)
        self.checkpoint_str = torch.load("model_result_56_b4_epoch_5_20231127_033430.pth", map_location=device)
        self.checkpoint_tum = torch.load("model_result_112_b2_epoch_6_20231127_043043.pth", map_location=device)
        
         

    def predict(self, imgs):
        # Initialize the model for multi-GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        transform_str = transforms.Compose([
            transforms.Resize((56, 56)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        inputs = self.transform(imgs).unsqueeze(0)
        inputs_str = transform_str(imgs).unsqueeze(0)
        inputs = inputs.to(device)
        
        model_val = EfficientNet.from_pretrained('efficientnet-b1')
        num_ftrs = model_val._fc.in_features
        model_val._fc = nn.Linear(num_ftrs, 9)
        model_val.to(device)
        model_val.load_state_dict(self.checkpoint_val)
        model_val.eval()
        
        with torch.no_grad():
            outputs = model_val(inputs)
            _, preds_val = torch.max(outputs, 1)
        
        model_tum = EfficientNet.from_pretrained('efficientnet-b2')
        num_ftrs = model_tum._fc.in_features
        model_tum._fc = nn.Linear(num_ftrs, 9)
        model_tum = nn.DataParallel(model_tum)
        model_tum.to(device)
        model_tum.load_state_dict(self.checkpoint_tum)
        model_tum.eval()
        with torch.no_grad():
            outputs_tum = model_tum(inputs)
            _, preds_tum = torch.max(outputs_tum, 1)
        
        model_str = EfficientNet.from_pretrained('efficientnet-b4')
        num_ftrs = model_str._fc.in_features
        model_str._fc = nn.Linear(num_ftrs, 9)
        model_str = nn.DataParallel(model_str)
        model_str.to(device)
        model_str.load_state_dict(self.checkpoint_str)
        model_str.eval()
        with torch.no_grad():
            outputs_str = model_str(inputs_str)
            _, preds_str = torch.max(outputs_str, 1)
        
        if preds_tum[0] == 8:
            return self.tag2class[preds_tum.cpu().numpy()[0]], self.tum_recall, self.tum_prec
        elif preds_str[0] == 7:
            return self.tag2class[preds_str.cpu().numpy()[0]], self.str_recall, self.str_prec
        else: 
            return self.tag2class[preds_val.cpu().numpy()[0]], self.val_acc, self.val_prec

def Macenko(image, target_image):
    image = np.array(image)
    target_image = np.array(target_image)
    n = stainNorm_Macenko.Normalizer()
    n.fit(target_image)
    image = n.transform(image)
    image = PIL.Image.fromarray(image)
    return image

m = ClassifyModel()
m.load()

st.sidebar.title("About")

st.sidebar.info(
    "This application identifies the tissue type in the picture.")


st.title('Histological Image Classification for Human Colorectal Cancer Detection')

st.write("Upload an image.")
uploaded_file = st.file_uploader(label=" ", label_visibility='collapsed')

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert('RGB')
    target_image = PIL.Image.open("TUM-TCGA-AAMWWECY.tif").convert('RGB')
    image = Macenko(image, target_image)
    tissue_type, accuracy, confidence = m.predict(image)
    st.write(f"I think this is **{tissue_type}** with accuracy: **{accuracy}** and confidence: **{round(float(confidence),4)*100}%**.")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
