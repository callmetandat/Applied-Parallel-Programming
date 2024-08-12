import numpy as np
import math
from torch import nn
from numba import jit, prange

class TransposeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0,weights=None, bias=None):
        super(TransposeConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding 

        # Initialize   
        self.weights = weights if weights is not None else np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = bias if bias is not None else np.zeros((out_channels,))
        
    @jit
    def forward(self, input_data):
        batch_size, in_channels, input_height, input_width = input_data.shape
        output_height = (input_height - 1) * self.stride + self.kernel_size + 2 * self.padding
        output_width = (input_width - 1) * self.stride + self.kernel_size + 2 * self.padding

        output = np.zeros((batch_size, self.out_channels, output_height, output_width))
        print(output.shape)
        for b in prange(batch_size):
            for c in prange(self.out_channels):
                for h in prange(output_height):
                    for w in prange(output_width):
                        h_in = int(h // self.stride)
                        w_in = int(w // self.stride)
                        h_kernel = int(h % self.kernel_size)
                        w_kernel = int(w % self.kernel_size)
                        output[b, c, h, w] = np.sum(input_data[b, :, h_in, w_in] * self.weights[:, c, h_kernel, w_kernel]) + self.bias[c]

        return output
    