from torch.nn import Module
from torch import Tensor
from torchvision import transforms
import torch
import numpy as np
from typing import Union
from model_numba.Layers.Layers import *
from numba import cuda

import torch.nn.functional as F

from numba.cuda.cudadrv.devicearray import DeviceNDArray

class Single_Conv(Module):
    def __init__(self, in_channel: int, out_channel: int, state_dict = None,prefix: str = "") -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if state_dict is not None:
            self.weight = state_dict[f'{prefix}.weight'].cpu().numpy()
            self.bias = state_dict[f'{prefix}.bias'].cpu().numpy()
        else:
            raise ValueError("State dictionary is required to initialize weights of double conv layer.")

    def forward(self, img: Union[np.ndarray, DeviceNDArray]) -> Union[np.ndarray, DeviceNDArray]:
        output = Conv_1D_Filter_GPU(img, self.weight, self.bias)
        return output


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
        
        del out_MaxPool, outConv_1

        return outConv_2
    
class Original_Double_Conv(Module):
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
        
        out_MaxPool = MaxPooling2D_GPU(img, True) if self.useMaxPooling else img

        outConv_1 = Convolution2D_GPU(out_MaxPool, self.conv1_weight, self.conv1_bias, True)
        outConv_1 = batchNorm2D(outConv_1, self.conv1_norm_weight, self.conv1_norm_bias, self.conv1_norm_mean, self.conv1_norm_var)
        outConv_1 = RELU_GPU(outConv_1)

        outConv_2 = Convolution2D_GPU(outConv_1, self.conv2_weight, self.conv2_bias, True)
        outConv_2 = batchNorm2D(outConv_2, self.conv2_norm_weight, self.conv2_norm_bias, self.conv2_norm_mean, self.conv2_norm_var)
        outConv_2 = RELU_GPU(outConv_2)
        del out_MaxPool, outConv_1

        return outConv_2
    
class Transpose_Conv2D(Module):
    def __init__(self, channel_in: int, channel_out: int, state_dict = None,prefix: str = "") -> None:
        super().__init__()

        self.channel_in = channel_in
        self.channel_out = channel_out

        if state_dict is not None:
            self.weight = state_dict[f'{prefix}.up.weight'].cpu().numpy()
            self.bias = state_dict[f'{prefix}.up.bias'].cpu().numpy()
        else:
            raise ValueError("State dictionary is required to initialize weights of transpose convolution.")
            

    def forward(self, img: Union[np.ndarray, DeviceNDArray]) -> Union[np.ndarray, DeviceNDArray]:
        if (self.channel_in == img.shape[1] and self.channel_out == self.weight.shape[1]):
            output = TransposeConvol2D_GPU(img, self.weight, self.bias, False)
            return output
        else:
            raise ValueError(f'Expected channel in and out: ({self.channel_in}, {self.channel_out}) but got ({img.shape[1], self.weight.shape[1]})')
        
class UP(Module):
    def __init__(self, in_channel: int, out_channel: int, state_dict = None,prefix: str = "") -> None:
        super().__init__()

        self.channel_in = in_channel
        self.channel_out = out_channel

        if state_dict is not None:
            self.weight = state_dict[f'{prefix}.up.weight'].cpu().numpy()
            self.bias = state_dict[f'{prefix}.up.bias'].cpu().numpy()


            self.conv1_weight = state_dict[f'{prefix}.double_conv.double_conv.0.weight'].cpu().numpy()
            self.conv1_bias = state_dict[f'{prefix}.double_conv.double_conv.0.bias'].cpu().numpy()
            self.conv1_norm_weight = state_dict[f'{prefix}.double_conv.double_conv.1.weight'].cpu().numpy()
            self.conv1_norm_bias = state_dict[f'{prefix}.double_conv.double_conv.1.bias'].cpu().numpy()
            self.conv1_norm_mean = state_dict[f'{prefix}.double_conv.double_conv.1.running_mean'].cpu().numpy()
            self.conv1_norm_var = state_dict[f'{prefix}.double_conv.double_conv.1.running_var'].cpu().numpy()

            self.conv2_weight = state_dict[f'{prefix}.double_conv.double_conv.3.weight'].cpu().numpy()
            self.conv2_bias = state_dict[f'{prefix}.double_conv.double_conv.3.bias'].cpu().numpy()
            self.conv2_norm_weight = state_dict[f'{prefix}.double_conv.double_conv.4.weight'].cpu().numpy()
            self.conv2_norm_bias = state_dict[f'{prefix}.double_conv.double_conv.4.bias'].cpu().numpy()
            self.conv2_norm_mean = state_dict[f'{prefix}.double_conv.double_conv.4.running_mean'].cpu().numpy()
            self.conv2_norm_var = state_dict[f'{prefix}.double_conv.double_conv.4.running_var'].cpu().numpy()
        else:
            raise ValueError("State dictionary is required to initialize weights of double conv layer.")

    def forward(self, img: Union[np.ndarray, DeviceNDArray], skip_img: Union[np.ndarray, DeviceNDArray], useTorchTranspose = False):
        if useTorchTranspose:
            input = torch.tensor(img).cuda()
            weight = torch.tensor(self.weight).cuda()
            bias = torch.tensor(self.bias).cuda()

            out_transpose = F.conv_transpose2d(input, weight, bias,stride= 2,padding= 0).cuda()
            out_transpose = out_transpose.cpu().detach().numpy()
        else:
            out_transpose = TransposeConvol2D_GPU(img, self.weight, self.bias, False)

        out_transpose = np.concatenate((out_transpose, skip_img), axis= 1)
        outConv_1 = Combie_TileConv_GPU(out_transpose, self.conv1_weight, self.conv1_bias, self.conv1_norm_weight,\
                                         self.conv1_norm_bias, self.conv1_norm_mean, self.conv1_norm_var, False)
        
        outConv_2 = Combie_TileConv_GPU(outConv_1, self.conv2_weight, self.conv2_bias, self.conv2_norm_weight,\
                                        self.conv2_norm_bias, self.conv2_norm_mean, self.conv2_norm_var, False)
        del out_transpose, outConv_1
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

        self.up_1 = UP(1024, 512, self.weights, 'up_1')
        self.up_2 = UP(512, 256, self.weights, 'up_2')
        self.up_3 = UP(256, 128, self.weights, 'up_3')
        self.up_4 = UP(128, 64, self.weights, 'up_4')

        self.out = Single_Conv(64, 1, self.weights, 'out')


    def forward(self, img: np.ndarray) -> np.ndarray:
        output =None
        with cuda.defer_cleanup():
            self.d1 = self.down_1(img)
            self.d2 = self.down_2(self.d1)
            self.d3 = self.down_3(self.d2)
            self.d4 = self.down_4(self.d3)
            self.d5 = self.down_5(self.d4)

            self.u1 = self.up_1(self.d5, self.d4, True)

            self.u2 = self.up_2(self.u1, self.d3, True)

            self.u3 = self.up_3(self.u2, self.d2, True)

            self.u4 = self.up_4(self.u3, self.d1, True)

            output = self.out(self.u4)

        return output