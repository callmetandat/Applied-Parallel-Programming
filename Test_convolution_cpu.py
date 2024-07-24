import math
import time
import numpy as np
import torch
import torch.nn as nn
from model_numba.layers_cpu.layer import *

BATCH_SIZE = 4
IN_CHANNELS = 64
OUT_CHANNELS = 128
IMG_HEIGHT = 128
IMG_WIDTH = 128
KERNEL_SIZE = 3

# Generate random input data
image = np.random.rand(BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)  # (batch_size, input_channels, height, width)
kernel = np.random.rand(OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)  # (output_channels, input_channels, kernel_height, kernel_width)

#print(image)
bias = np.random.rand(OUT_CHANNELS).astype(np.float32)

start_time = time.time()
conv_layer = ConvolutionalLayer(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2, weights=kernel, bias=bias)  # Create a Conv instance
output = conv_layer.forward(image)
#print(output)
print("--- %s seconds ---" % (time.time() - start_time))


torch_img = torch.tensor(image, requires_grad=False).cpu()
torch_weight = torch.tensor(kernel, requires_grad=False).cpu()
torch_bias = torch.tensor(bias, requires_grad=False).cpu()

conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2, bias=True).cpu()
conv.weight.data = torch_weight
conv.bias.data = torch_bias
start = time.time()
torch_output = conv(torch_img).cpu().detach().numpy()
print("--- %s seconds ---" % (time.time() - start))


# Check for correctness
difference = np.abs(output - torch_output)
print("Max difference:", np.max(difference))
print("Mean difference:", np.mean(difference))
print("Sample differences:\n", difference[0, 0, :5, :5])