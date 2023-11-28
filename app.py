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
import numpy as np
import PIL


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
        self.val_acc = 0.9568245125348189
        self.str_acc = 0.7648456057007126
        self.tum_acc = 0.9878345498783455
        

    def load(self):
        image_size = 112
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.checkpoint_val = torch.load("model_result_112_b3_epoch_4_20231127_052702.pth", map_location=device)
        self.checkpoint_str = torch.load("model_result_112_b3_epoch_1_20231127_184058.pth", map_location=device)
        self.checkpoint_tum = torch.load("model_result_112_b3_epoch_4_20231127_185353.pth", map_location=device)
        
         

    def predict(self, imgs):
        # Initialize the model for multi-GPU if available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = EfficientNet.from_pretrained('efficientnet-b3')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, 9)
        model = nn.DataParallel(model)
        model.to(device)

        inputs = self.transform(imgs).unsqueeze(0)
        inputs = inputs.to(device)
        model.load_state_dict(self.checkpoint_val)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            _, preds_val = torch.max(outputs, 1)
        model.load_state_dict(self.checkpoint_str)
        model.eval()
        with torch.no_grad():
            outputs_str = model(inputs)
            _, preds_str = torch.max(outputs_str, 1)
        model.load_state_dict(self.checkpoint_tum)
        model.eval()
        with torch.no_grad():
            outputs_tum = model(inputs)
            _, preds_tum = torch.max(outputs_tum, 1)
        if preds_tum[0] == 8:
            return self.tag2class[preds_tum.cpu().numpy()[0]], self.tum_acc
        elif preds_str[0] == 7:
            return self.tag2class[preds_str.cpu().numpy()[0]], self.str_acc
        else: 
            return self.tag2class[preds_val.cpu().numpy()[0]], self.val_acc

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
    # img = np.array(image)
    tissue_type, confidence = m.predict(image)
    st.write(f"I think this is **{tissue_type}**(confidence: **{round(float(confidence),4)*100}%**)")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
