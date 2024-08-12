import numpy as np
import time
import torch
import torch.nn as nn
from model_numba.Layers.Layers import Split_Channel_TileSharedConv2D_GPU, Stream_TileSharedConv2D_GPU, Convolution2D_GPU
import sys

from model_numba.layers_cpu.layer import ConvolutionalLayer
# Define constants for testing
BATCH_SIZE = int(sys.argv[1])
IN_CHANNELS = int(sys.argv[2])
OUT_CHANNELS = int(sys.argv[3])
IMG_HEIGHT = int(sys.argv[4])
IMG_WIDTH = int(sys.argv[4])
KERNEL_SIZE = 3

# Generate random input data
image = np.random.uniform(-1.0, 1.0, (BATCH_SIZE, IN_CHANNELS, IMG_HEIGHT, IMG_WIDTH)).astype(np.float32)
kernel = np.random.uniform(0.0, 1.0, (OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE)).astype(np.float32)
bias = np.random.uniform(0.0, 1.0, OUT_CHANNELS).astype(np.float32)

# Prepare PyTorch tensors
torch_img = torch.tensor(image, requires_grad=False).cuda()
torch_weight = torch.tensor(kernel, requires_grad=False).cuda()
torch_bias = torch.tensor(bias, requires_grad=False).cuda()

# Define and measure PyTorch convolution runtime
conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, padding=KERNEL_SIZE // 2, bias=True).cuda()
conv.weight.data = torch_weight
conv.bias.data = torch_bias

dr_conv = ConvolutionalLayer(IN_CHANNELS,OUT_CHANNELS, KERNEL_SIZE, weights= kernel, bias= bias, padding= 1)

# Measure the runtime and check for correctness
times = []

start = time.time()
#cuda_out = dr_conv.forward(image)
end = time.time()
cpu_time = end - start
print("Direct CPU time:", cpu_time)


start = time.time()
cuda_out = Convolution2D_GPU(image, kernel, bias)
end = time.time()
shared_time = end - start
print("TileSharedConv2D_GPU time:", shared_time)

start = time.time()
torch_output = conv(torch_img)
end = time.time()
torch_time = end - start
print("PyTorch Conv2D time:", torch_time)

torch_output = torch_output.cpu().detach().numpy()

# Compare results
difference_shared = np.abs(cuda_out - torch_output)
print("Max difference (Shared):", np.max(difference_shared))
print("Mean difference (Shared):", np.mean(difference_shared))


 