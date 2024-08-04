from torch.nn import Module
from torch import Tensor
from torchvision import transforms
import torch
import numpy as np
from typing import Union
from model_numba.Layers.Layers import *

from numba.cuda.cudadrv.devicearray import DeviceNDArray

class Double_Conv(Module):
    def __init__(self, in_channel: int, out_channel: int, state_dict = None,prefix: str = "", isMaxPooling: bool = False) -> None:
        super().__init__()

        self.useMaxPooling = isMaxPooling
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        if state_dict is not None:
            self.conv1_weight = state_dict[f'{prefix}.double_conv.0.weight'].cpu().numpy()
            self.conv1_bias = state_dict[f'{prefix}.double_conv.0.bias'].cpu().numpy()
            self.conv1_norm_weight = state_dict[f'{prefix}.double_conv.1.weight'].cpu().numpy()
            self.conv1_norm_bias = state_dict[f'{prefix}.double_conv.1.bias'].cpu().numpy()
            self.conv1_norm_mean = state_dict[f'{prefix}.double_conv.1.running_mean'].cpu().numpy()
            self.conv1_norm_var = state_dict[f'{prefix}.double_conv.1.running_var'].cpu().numpy()

            self.conv2_weight = state_dict[f'{prefix}.double_conv.3.weight'].cpu().numpy()
            self.conv2_bias = state_dict[f'{prefix}.double_conv.3.bias'].cpu().numpy()
            self.conv2_norm_weight = state_dict[f'{prefix}.double_conv.4.weight'].cpu().numpy()
            self.conv2_norm_bias = state_dict[f'{prefix}.double_conv.4.bias'].cpu().numpy()
            self.conv2_norm_mean = state_dict[f'{prefix}.double_conv.4.running_mean'].cpu().numpy()
            self.conv2_norm_var = state_dict[f'{prefix}.double_conv.4.running_var'].cpu().numpy()
        else:
            raise ValueError("State dictionary is required to initialize weights of double conv layer.")

    def forward(self, img: Union[np.ndarray, DeviceNDArray]) -> Union[np.ndarray, DeviceNDArray]:
        
        out_MaxPool = MaxPooling2D_GPU(img, False) if self.useMaxPooling else img

        outConv_1 = Combie_TileConv_GPU(out_MaxPool, self.conv1_weight, self.conv1_bias, self.conv1_norm_weight,\
                                         self.conv1_norm_bias, self.conv1_norm_mean, self.conv1_norm_var, False)
        
        outConv_2 = Combie_TileConv_GPU(outConv_1, self.conv2_weight, self.conv2_bias, self.conv2_norm_weight,\
                                        self.conv2_norm_bias, self.conv2_norm_mean, self.conv2_norm_var, False)
        
        return outConv_2


class Unet_Cuda(Module):
    def __init__(self, num_classes: int, weight_path: str = r'weights\weights.pth') -> None:
        super().__init__()

        self.weights = torch.load(weight_path)

        self.down_1 = Double_Conv(3, 64, self.weights, 'down_1', False)
        self.down_2 = Double_Conv(64, 128, self.weights, 'down_2.max_pooling_and_conv.1', True)
        self.down_3 = Double_Conv(128, 256, self.weights, 'down_3.max_pooling_and_conv.1', True)
        self.down_4 = Double_Conv(256, 512, self.weights, 'down_4.max_pooling_and_conv.1', True)
        self.down_5 = Double_Conv(512, 1024, self.weights, 'down_5.max_pooling_and_conv.1', True)


    def forward(self, img: np.ndarray) -> np.ndarray:
        self.d1 = self.down_1(img)
        self.d2 = self.down_2(self.d1)
        self.d3 = self.down_3(self.d2)
        self.d4 = self.down_4(self.d3)
        self.d5 = self.down_5(self.d4)

        return self.d5