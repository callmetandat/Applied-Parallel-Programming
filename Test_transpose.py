import math
import time
import numpy as np
import torch
import torch.nn as nn
from model_numba.layers_cpu.transpose import *

BATCH_SIZE = 1
IN_CHANNELS = 128
OUT_CHANNELS = 64
IMG_HEIGHT = 256
IMG_WIDTH = 256
KERNEL_SIZE = 2

# Generate random input data
image = np.random.rand(BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)  # (batch_size, input_channels, height, width)
kernel = np.random.rand(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)  # (output_channels, input_channels, kernel_height, kernel_width)
bias = np.random.rand(OUT_CHANNELS).astype(np.float32)

# Conv
torch_img = torch.tensor(image, requires_grad=False).cpu()
torch_weight = torch.tensor(kernel, requires_grad=False).cpu()
torch_bias = torch.tensor(bias, requires_grad=False).cpu()

start_me = time.time()
transpose_conv = TransposeConvLayer(IN_CHANNELS, OUT_CHANNELS, 2, weights=kernel, bias=bias)
output = transpose_conv.forward(image) 
end_me = time.time()

start_torch = time.time()
conv = nn.ConvTranspose2d(IN_CHANNELS, OUT_CHANNELS, 2, stride=2, bias=True).cpu()
conv.weight.data = torch_weight
conv.bias.data = torch_bias
torch_output = conv(torch_img).cpu().detach().numpy()
end_torch = time.time()

print("--- My time: %s seconds ---" % (end_me - start_me))
print("--- Torch time: %s seconds ---" % (end_torch - start_torch))

# Check for correctness
difference = np.abs(output - torch_output)
print("Max difference:", np.max(difference))
print("Mean difference:", np.mean(difference))
print("Sample differences:\n", difference[0, 0, :5, :5])