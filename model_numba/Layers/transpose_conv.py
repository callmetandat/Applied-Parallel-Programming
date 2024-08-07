from numba import cuda
import numba
import numpy as np
import math
from typing import Union

BLOCK_SIZE = 16
KERNEL_SIZE = 2
#KERNEL_SIZE_AS_1D_ARR = KERNEL_SIZE * KERNEL_SIZE
#INPUT_TILE_SIZE = BLOCK_SIZE
#OUTPUT_TILE_SIZE = BLOCK_SIZE - KERNEL_SIZE + 1

@cuda.jit
def TransposeConv2D_kernel(in_img, out_img, weight, bias, channel_in: int, channel_out: int):
    """
    
    """
    _, _, z_idx = cuda.grid(3)
    col, row = cuda.threadIdx.x, cuda.threadIdx.y
    
    start_col = col * 2 + KERNEL_SIZE // 2
    start_row =  row * 2 + KERNEL_SIZE // 2
    
    batch_idx, out_channel_idx = z_idx // channel_out, z_idx % channel_out
    
    input_height, input_width = in_img[2], in_img[3]
    output_height, output_width = out_img[2], out_img[3]
    outPixel = bias[out_channel_idx]
    
    for in_channel_idx in range(channel_in):
        for r in range(KERNEL_SIZE):
            for c in range(KERNEL_SIZE):
                if(start_row + r >=0 and start_row + r < input_height and start_col + c >= 0 and start_col + c < input_width):
                    res += in_img[batch_idx, in_channel_idx, r, c] * weight[out_channel_idx, in_channel_idx, r, c]
        cuda.syncthreads()
        
    out_img[batch_idx, out_channel_idx, start_row, start_col]
                        
    
        
def TransposeConvol2D_GPU(img: Union[np.ndarray, any], weight, bias):
    IMG_WIDTH, IMG_HEIGHT = img.shape[3], img.shape[2]
    BATCH_SIZE = img.shape[0]
    IN_CHANNEL, OUT_CHANNEL = weight.shape[1], weight.shape[0]

    threadPerBlock = (BLOCK_SIZE, BLOCK_SIZE)
    blockPerGrid = (math.ceil(IMG_WIDTH), math.ceil(IMG_HEIGHT), BATCH_SIZE * OUT_CHANNEL)
    with cuda.pinned(img, weight, bias):
        d_img = cuda.to_device(img)
        d_out_img = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, IMG_HEIGHT, IMG_WIDTH), dtype= img.dtype)
        d_weight = cuda.to_device(weight)
        d_bias = cuda.to_device(bias)

        TransposeConv2D_kernel[blockPerGrid, threadPerBlock](d_img, d_out_img, d_weight, d_bias, IN_CHANNEL, OUT_CHANNEL, BATCH_SIZE)

        out_img = d_out_img.copy_to_host()

    return out_img
    
 