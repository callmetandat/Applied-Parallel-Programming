from torch.nn import Module
from torch import Tensor
from torchvision import transforms
import torch
import numpy as np

from model_numba.Layers import *


class TileConv(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = Convolution2D_GPU

    def forward(self, img, weight, bias):
        return self.conv(img, weight, bias)

class SplitTileConv(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = Split_Channel_TileSharedConv2D_GPU

    def forward(self, img, weight, bias):
        return self.conv(img, weight, bias, 64)

class BatchNorm2D(Module):
    def __init__(self) -> None:
        super().__init__()
        self.batchNorm = ...

    def forward(self, img, weight, bias, var):
        return self.batchNorm(img, weight, bias, var)
    
class MaxPooling(Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpooling = ...

    def forward(self, img):
        return self.maxpooling(img)
    
class CombieTileConv(Module):
    pass

    


kernel_list = ['basic_conv', 'tilesharedconv', 'splittileconv', 'combie_tile_sharedconv', 'combie_spiltchannelconv']
class Double_Conv(Module):
    def __init__(self, in_channel: int, out_channel: int, mode: str = 'spiltchannelconv') -> None:
        super().__init__()
        if mode not in kernel_list:
            raise 'Kernel is invalid'
        
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, img: np.ndarray):
        ...






class Unet_Cuda(Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.isWeightLoaded = False

    def forward(self, img: Tensor) -> Tensor:
        pass

    def load_weight(self,path: str) -> None:
        self.weight = torch.load(path)
        



