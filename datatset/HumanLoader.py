from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch import Tensor
import json
import os
import numpy as np

class HumanMattingDataset(Dataset):
    def __init__(self, data_dir = 'dataset', split = 'train') -> None:
        self.data_dir = data_dir

        self.transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor()
            ]
        )
        
        self.img_path = os.path.join(data_dir, 'clip_img')
        self.mask_path = os.path.join(data_dir,'matting')

        mode = 'train' if split == 'train' else 'test'

        self.img_list = []

        #Load Img List From Json File
        with open(os.path.join(data_dir,mode + '.json'), 'r') as f:
            self.img_list =json.loads(f.read())

    def get_img_list(self) -> list:
        return self.img_list

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index):
        img_name: str = self.img_list[index]

        #Load Image
        image = Image.open(self.img_path +"\\"+ img_name)
        image = self.transform(image)

        #Load matting and mask
        matting = Image.open(self.mask_path + "\\"+ (img_name[:-3]+ 'png').replace('clip','matting'))
        mask = matting.copy()

        alpha = mask.split()[-1]
        mask = alpha.point(lambda p: 255 if p > 127 else 0)
        #matting = np.array(matting)
        matting = self.transform(matting)
        mask = self.transform(mask)
        #mask= mask[:,:,3]
        return image, mask, matting, img_name