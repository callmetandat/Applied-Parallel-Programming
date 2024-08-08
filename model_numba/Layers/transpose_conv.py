from numba import cuda
import numba
import numpy as np
import math
from typing import Union

BLOCK_SIZE = 16
KERNEL_SIZE = 2
STRIDE = 2

@cuda.jit
def TransposeConv2D_kernel(in_img, out_img, weight, bias, channel_in: int, channel_out: int, batch_size: int):
    """
    
    """
    col, row, z_idx = cuda.grid(3)
    #col, row = cuda.threadIdx.x, cuda.threadIdx.y
    
    output_height, output_width = out_img.shape[2], out_img.shape[3]
    
    #block_col = cuda.blockIdx.x * BLOCK_SIZE
    #block_row = cuda.blockIdx.y 
    start_col = cuda.threadIdx.x * STRIDE - (KERNEL_SIZE // 2)
    start_row = cuda.threadIdx.y * STRIDE - (KERNEL_SIZE // 2)
    
    batch_idx, out_channel_idx = z_idx // channel_out, z_idx % channel_out
    
    outPixel = bias[out_channel_idx]
    
    if (col <= output_width) and (row <= output_height):  
        for in_channel_idx in range(channel_in):
            #if((start_row >= 0) and (start_row < output_height) and (start_col >= 0) and (start_col < output_width)):
            outPixel += in_img[batch_idx, in_channel_idx, start_row, start_col] * weight[in_channel_idx, out_channel_idx, start_row, start_col] 
       
    out_img[batch_idx, out_channel_idx, start_row, start_col] = outPixel
                                
def TransposeConvol2D_GPU(img: Union[np.ndarray, any], weight, bias):
    IMG_WIDTH, IMG_HEIGHT = img.shape[3], img.shape[2]
    BATCH_SIZE = img.shape[0]
    IN_CHANNEL, OUT_CHANNEL = weight.shape[0], weight.shape[1]

    threadPerBlock = (BLOCK_SIZE, BLOCK_SIZE)
    blockPerGrid = (math.ceil(IMG_WIDTH), math.ceil(IMG_HEIGHT), BATCH_SIZE * OUT_CHANNEL)
    with cuda.pinned(img, weight, bias):
        d_img = cuda.to_device(img)
        d_out_img = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, IMG_HEIGHT*2, IMG_WIDTH*2), dtype= img.dtype)
        d_weight = cuda.to_device(weight)
        d_bias = cuda.to_device(bias)

        TransposeConv2D_kernel[blockPerGrid, threadPerBlock](d_img, d_out_img, d_weight, d_bias, IN_CHANNEL, OUT_CHANNEL, BATCH_SIZE)

        out_img = d_out_img.copy_to_host()

    return out_img
    
 