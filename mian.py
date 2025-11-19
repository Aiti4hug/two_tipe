from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import streamlit as st
import os

#gray
transform_data_g = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

#rgb
transform_data_r = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


#gray
class CarCycG(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


#rgb
class CarCycR(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


check_image_app = FastAPI()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

carcyc_gray = CarCycG().to(device)
carcyc_rgb = CarCycR().to(device)

carcyc_gray.load_state_dict(torch.load('models/carcyc_gray.pth', map_location=device))
carcyc_gray.eval()

carcyc_rgb.load_state_dict(torch.load('models/carcyc_rgb.pth', map_location=device))
carcyc_rgb.eval()


classes = [
    'car',
    'motocycle'
]

st.title('FIVE TYPE CLASSIFIER MODEL')
st.text('LOAD image')

type_image = st.file_uploader('drop images', type=['png', 'jpg', 'jpeg'])

if type_image is not None:
    st.image(type_image, caption='loader')

model_type = st.selectbox(
    "Choose model type",
    ["GRAY model", "RGB model"]
)

if st.button('Determine the number'):
    try:
        if type_image is None:
            st.error("Please upload an image first!")
            st.stop()

        image = Image.open(type_image).convert("RGB")

        if model_type == "GRAY model":
            tensor = transform_data_g(image).unsqueeze(0).to(device)
            model = carcyc_gray
        else:
            tensor = transform_data_r(image).unsqueeze(0).to(device)
            model = carcyc_rgb

        with torch.no_grad():
            pred = model(tensor).argmax(1).item()

        st.success(f"Answer: {classes[pred]}")

    except Exception as e:
        st.error(f"Error: {str(e)}")
