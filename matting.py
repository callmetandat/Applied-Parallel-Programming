import cv2
import torch
from PIL import Image
import streamlit as st
from torchvision import transforms
from model.unet import U_net

@st.cache_resource
def load_model():
    model = U_net(1).to(device='cpu')
    weights_dict = torch.load(r'weights/unet_model_best.pth', map_location=torch.device(device='cpu'))

    # Load weights into the model
    model.load_state_dict(weights_dict)
    return model

@st.cache_data    
def load_img(file_path, mask:bool=False):
    img = cv2.imread(file_path)
    if(mask):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_mask(img, model):
    transform = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Resize((304, 512))
        ]
    )
    x = transform(img)
    x = x.unsqueeze(0)
    x.shape
    
    pred = model(x.repeat(1,1,1,1).cpu())
    x = (pred.detach() > 0.1).float()
    x = x.cpu()
    
    np_images = x.numpy().transpose((0, 1, 2, 3))
    np_images = np_images.squeeze()
    return np_images

@st.cache_data
def change_pixel(img, bg, mask):
    res = img.copy()
    # Resize background and mask to match with the portrait img
    bg = cv2.resize(bg, (res.shape[1], res.shape[0]))
    mask = cv2.resize(mask, (res.shape[1], res.shape[0]))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Background pixels of binary mask have value of 0
            if mask[i, j] == 0:
                res[i, j] = bg[i, j]
    return res