import math
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from typing import Union
import torch.nn.functional as F
import torch
# Assuming BLOCK_SIZE_FC is defined somewhere in your environment
BLOCK_SIZE_FC = 256  # Example block size, adjust as needed

@cuda.jit
def Conv_1D_Filter_kernel(img: Union[np.ndarray, DeviceNDArray], out_img: Union[np.ndarray, DeviceNDArray],
                          weight: DeviceNDArray, bias: DeviceNDArray, in_channel: int):
    x_idx, batch_idx = cuda.grid(2)
    if x_idx < img.shape[2] * img.shape[3]:
        row, col = x_idx // img.shape[3], x_idx % img.shape[3]

        outPixel = bias[0]
        for i in range(in_channel):
            outPixel += weight[0, i, 0, 0] * img[batch_idx, i, row, col]

        out_img[batch_idx, 0, row, col] = outPixel

def Conv_1D_Filter_GPU(img: Union[np.ndarray, DeviceNDArray], weight: np.ndarray, bias: np.ndarray):
    BATCH_SIZE, IN_CHANNEL, HEIGHT, WIDTH = img.shape
    OUT_CHANNEL = 1

    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)
    d_img = cuda.to_device(img) if isinstance(img, np.ndarray) else img

    d_output = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, HEIGHT, WIDTH), dtype=img.dtype)

    threadPerBlock = (BLOCK_SIZE_FC,)
    blockPerGrid = (math.ceil(HEIGHT * WIDTH / BLOCK_SIZE_FC), BATCH_SIZE)

    Conv_1D_Filter_kernel[blockPerGrid, threadPerBlock](d_img, d_output, d_weight, d_bias, IN_CHANNEL)

    output = d_output.copy_to_host()
    return output

# Example usage
BATCH_SIZE = 4
IN_CHANNEL = 64
HEIGHT = 512
WIDTH = 512
img = np.random.rand(BATCH_SIZE, IN_CHANNEL, HEIGHT, WIDTH).astype(np.float32)
weight = np.random.rand(1, IN_CHANNEL, 1, 1).astype(np.float32)
bias = np.random.rand(1).astype(np.float32)

output_numba = Conv_1D_Filter_GPU(img, weight, bias)



input_torch = torch.tensor(img)  # Convert to torch tensor
weight_torch = torch.tensor(weight)
bias_torch = torch.tensor(bias)

# Use a 2D convolution with (1,1) kernel size to simulate a 1D filter over each channel
output_torch = F.conv2d(input_torch, weight_torch, bias=bias_torch)

# Convert PyTorch output to numpy and compare
output_torch_np = output_torch.detach().numpy()

# Check if the outputs are close
print("Outputs are close:", np.allclose(output_numba, output_torch_np, atol=1e-5))

# Print outputs for comparison if needed
print("Numba Output:\n", output_numba[0, 0, :, :])
print("PyTorch Output:\n", output_torch_np[0, 0, :, :])
