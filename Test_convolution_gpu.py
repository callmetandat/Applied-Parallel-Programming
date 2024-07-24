import numpy as np
from model_numba.layers_cuda.layer import Convolution2D_GPU
import math

# Define constants for testing
BATCH_SIZE = 4
IN_CHANNELS = 64
OUT_CHANNELS = 128
IMG_HEIGHT = 512
IMG_WIDTH = 256
KERNEL_SIZE = 3

# Generate random input data
image = np.random.rand(BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)  # (batch_size, input_channels, height, width)
kernel = np.random.rand(OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE).astype(np.float32)  # (output_channels, input_channels, kernel_height, kernel_width)
bias = np.random.rand(OUT_CHANNELS).astype(np.float32)

cuda_output = Convolution2D_GPU(image, kernel, bias)

import torch
import torch.nn as nn

torch_img = torch.tensor(image, requires_grad=False).cuda()
torch_weight = torch.tensor(kernel, requires_grad=False).cuda()
torch_bias = torch.tensor(bias, requires_grad=False).cuda()

conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2, bias=True).cuda()
conv.weight.data = torch_weight
conv.bias.data = torch_bias
torch_output = conv(torch_img).cpu().detach().numpy()

# Check for correctness
difference = np.abs(cuda_output - torch_output)
print("Max difference:", np.max(difference))
print("Mean difference:", np.mean(difference))
print("Sample differences:\n", difference[0, 0, :5, :5])