from torch.nn import Module
from torch import Tensor
from torchvision import transforms

from model_numba.Layers import *

class Unet_Cuda(Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

    def forward(self, img):
        pass