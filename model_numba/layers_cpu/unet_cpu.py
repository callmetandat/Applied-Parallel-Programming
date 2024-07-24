import torch 
from layer import *

class Unet():
    def __init__(self):
        self.conv1 = ConvolutionalLayer(nb_filters= 3, filter_size= 3, nb_channels= 3)
        
        