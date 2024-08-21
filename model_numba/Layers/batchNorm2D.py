import numpy as np
from numba import cuda, float32
import math

@cuda.jit
def batchNorm2D_kernel(img, out_img, batchNorm_weight, batchNorm_bias, mean, variance, epsilon):
    out_x, out_y = cuda.grid(2)

    in_channel_idx = cuda.blockIdx.z % img.shape[1]
    batch_idx = cuda.blockIdx.z // img.shape[1]

    if out_x < img.shape[3] and out_y < img.shape[2] and cuda.blockIdx.z < img.shape[0] * img.shape[1]:
        norm_val = (img[batch_idx, in_channel_idx, out_y, out_x] - mean[in_channel_idx]) / math.sqrt(variance[in_channel_idx] + epsilon)
        out_img[batch_idx, in_channel_idx, out_y, out_x] = batchNorm_weight[in_channel_idx] * norm_val + batchNorm_bias[in_channel_idx]

def batchNorm2D(img, weight, bias, mean, variance, epsilon=1e-5):
    batch_size, channels, height, width = img.shape
    out_img = np.zeros_like(img, dtype=np.float32)

    # Allocate device memory
    d_img = cuda.to_device(img)
    d_out_img = cuda.device_array_like(img)
    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)
    d_mean = cuda.to_device(mean)
    d_variance = cuda.to_device(variance)

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(width / threadsperblock[0])
    blockspergrid_y = math.ceil(height / threadsperblock[1])
    blockspergrid_z = batch_size * channels
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Launch kernel
    batchNorm2D_kernel[blockspergrid, threadsperblock](d_img, d_out_img, d_weight, d_bias, d_mean, d_variance, epsilon)

    # Copy result back to host
    out_img = d_out_img.copy_to_host()

    return out_img
