from fastapi import FastAPI, UploadFile, File, HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

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

@check_image_app.post("/predict/gray")
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="n0 file")

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform_data_g(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = carcyc_gray(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": classes[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@check_image_app.post("/predict/rgb")
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="n0 file")

        img = Image.open(io.BytesIO(image_data))
        img_tensor = transform_data_r(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = carcyc_rgb(img_tensor)
            pred = y_pred.argmax(dim=1).item()

        return {"Answer": classes[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(check_image_app, host='127.0.0.1', port=8000)