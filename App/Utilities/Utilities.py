import torchvision.transforms.transforms as transforms
import streamlit as st
import os
import numpy as np
import pickle as pk
import torch.nn as nn
import torch
class PneumoniaClasifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_Sequential = nn.Sequential( # input : 200 * 200 * 1
            nn.Conv2d(1 , 30 , 3 , stride = 2), # output : 99 * 99 * 30
            nn.ReLU(),
            nn.Conv2d(30 , 20 , 3 , stride = 2), # output : 49 * 49 * 20
            nn.ReLU(),
            nn.MaxPool2d(3 , stride = 1), # output : 47 * 47 * 20
            nn.Conv2d(20 , 30 , 3 , stride = 1), # output : 45 * 45 * 30
            nn.ReLU(),
            nn.Conv2d(30 , 20 , 3 , stride = 1), # output : 43 * 43 * 20
            nn.ReLU(),
            nn.MaxPool2d(3 , stride = 1) # output : 41 * 41 * 20
        )
        
        self.Flatten = nn.Flatten()
        
        self.FC = nn.Sequential(
            nn.Linear(33620 , 100),
            nn.ReLU(),
            nn.Linear(100 ,50),
            nn.ReLU(),
            nn.Linear(50 , 20),
            nn.ReLU(),
            nn.Linear(20 , 1),
            nn.Sigmoid()
        )

    def forward(self , x):
        x = self.Conv_Sequential(x)
        
        x = self.Flatten(x)
       
        x = self.FC(x)

        return x

utilities_current_dir = "/".join(os.path.dirname(__file__).split("/")[ :-1])

model_path = os.path.join(utilities_current_dir , "model" , "Pneumonia_model.pk")

def upload_css_file(assets_dir , file_name):
    file_path = os.path.join(assets_dir , file_name)
    with open(file_path , "r") as f:
        css_content = f.read()
        st.html(f"<style>{css_content}</style>")

def reorder_channels(image : np.ndarray):
    new_image = image.reshape((image.shape[1] , image.shape[2] , image.shape[0]))

    return new_image

def load_model(model_path = model_path):
    with open(model_path , 'br') as f:
        model = pk.load(f)
        return model
    

class Model:
    def __init__(self , model):
        self.model = model
        self.transform = transforms.Compose([
            # PIL tranformations
            transforms.Grayscale(1),
            transforms.Pad(150),
            transforms.Resize(200),
            transforms.CenterCrop(200),

            # Tranforming PIL to tensor
            transforms.ToTensor(),
            
            # Numpy transofmations
            
        ])

    def predict(self , image : torch.Tensor, threshold = 0.5):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        outputs = self.model(image)
        prediction = (outputs >= threshold).to(torch.int).squeeze().item()

        if prediction == 1:
            return 'Pneumonia'
        
        return 'Normal'
    
