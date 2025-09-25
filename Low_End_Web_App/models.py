from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.cuda.amp import autocast,GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.4)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 128))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) 
        
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128) 
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64) 
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(64, 32) 
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1) 
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits

MODEL_PATH = "C:\\Projects 2\\Biofouling\\Models\\best_model_dice_0.5029.pth"
IN_CHANNELS = 3
OUT_CHANNELS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SegModel = UNet(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS)

SegModel.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

SegModel.to(DEVICE)
SegModel.eval()

optimizer = torch.optim.Adam(SegModel.parameters(), lr=1e-6)


IMG_HEIGHT = 256
IMG_WIDTH = 256
preprocess = T.Compose([
    T.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def PredictSegment(In="testing/rust.png"):
    input_image = Image.open(In).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) 
    print(f"Shape before fix: {input_tensor.shape}")
    print(f"Shape after fix: {input_batch.shape}")

    with torch.no_grad():
        prediction = SegModel(input_batch.to(DEVICE))


    print("\n✅ Prediction successful!")
    prediction_mask_tensor = torch.argmax(prediction.squeeze(), dim=0)

    prediction_mask_numpy = prediction_mask_tensor.cpu().numpy() #OUTPUT 1
    prediction_mask_numpy[prediction_mask_numpy==1]=3.
    
    # Load and ensure RGB format
    img_data = plt.imread(In)
    if img_data.shape[-1] == 4:  # RGBA
        img_data = img_data[:,:,:3]  # Convert to RGB
    half = cv2.resize(img_data, (256, 256), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
    
    print(prediction_mask_numpy[0][100])
    for i in range(len(prediction_mask_numpy)):
        for y in range(len(prediction_mask_numpy[0])):
            if prediction_mask_numpy[i][y]!=0:
                half[i][y]=np.array([1,1,1],dtype="uint8")**prediction_mask_numpy[i][y]


    a=0
    b=0
    for i in range(len(prediction_mask_numpy)):
        for y in range(len(prediction_mask_numpy[0])):
            if prediction_mask_numpy[i][y]!=0:
                a+=1
            else:
                b+=1


    w, h = 256,256
    edge = [[0 for x in range(w)] for y in range(h)] 
    k=100
    for i in range(1,len(prediction_mask_numpy)-1):
        for y in range(1,len(prediction_mask_numpy[0])-1):
            if prediction_mask_numpy[i][y]!=0 and (prediction_mask_numpy[i][y+1]==0 or prediction_mask_numpy[i][y-1]==0):
                edge[i][y]=prediction_mask_numpy[i][y]*k
                edge[i][y+1]=prediction_mask_numpy[i][y]*k
                edge[i][y-1]=prediction_mask_numpy[i][y]*k
            if prediction_mask_numpy[i][y]!=0 and (prediction_mask_numpy[i+1][y]==0 or prediction_mask_numpy[i-1][y]==0):
                edge[i][y]=prediction_mask_numpy[i][y]*k
                edge[i+1][y]=prediction_mask_numpy[i][y]*k
                edge[i-1][y]=prediction_mask_numpy[i][y]*k

    # Load and ensure RGB format for half2
    img_data2 = plt.imread(In)
    if img_data2.shape[-1] == 4:  # RGBA
        img_data2 = img_data2[:,:,:3]  # Convert to RGB
    half2 = cv2.resize(img_data2, (256, 256), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

    for i in range(len(edge)):
        for y in range(len(edge[0])):
            if edge[i][y]!=0:
                half2[i][y]=np.array([1,1,1],dtype="uint8")*edge[i][y]
    return half,half2,prediction_mask_numpy,a/(a+b)

num_classes=11
IMG_SIZE=256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
class MultiLabelCNN(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        dummy_output = self.features(dummy_input)
        n_features = dummy_output.flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = MultiLabelCNN(num_classes).to(DEVICE)
print("\nModel Architecture:")
print(model)
model.load_state_dict(torch.load("C:\\Projects 2\\Biofouling\\classimodel1\\best_biofouling_model.pth", map_location=DEVICE))

model.to(DEVICE)
model.eval()
preprocess = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def PredictSpecies(inn="testing/rust.png"):
    input_image = Image.open(inn).convert("RGB")
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        pred = model(input_batch.to(DEVICE))
    return ["Algae","Barnacles","Clean","Hydrozoan","Jellyfish","Mussels","Rust","Starfish","Worms","Zebra Mussels","Tunicates"][torch.argmax(pred)]


if __name__ == "__main__":
    a,b,c,d=PredictSegment()
    plt.imshow(a)
    plt.imshow(b)
    plt.imshow(c)
    print(d)
    print(PredictSpecies())
