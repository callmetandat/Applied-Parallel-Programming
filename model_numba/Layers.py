from numba import cuda
import numba
import numpy as np
import math

BLOCKSIZE = 16
KERNEL_SIZE = 3
OUTPUT_TILE_SIZE = BLOCKSIZE - KERNEL_SIZE + 1
INPUT_TILE_SIZE = BLOCKSIZE

@cuda.jit
def tileSharedConv2D_kernel(img, out_img, weight, bias, channel_in: int, channel_out: int, batch_size: int):
    _, _, z_idx = cuda.grid(3)

    batch_idx, out_channel_idx = z_idx // channel_out, z_idx % channel_out

    shared_col, shared_row = cuda.threadIdx.x, cuda.threadIdx.y

    col_out = cuda.blockIdx.x * OUTPUT_TILE_SIZE + shared_col
    row_out = cuda.blockIdx.y * OUTPUT_TILE_SIZE + shared_row

    col_in = col_out - KERNEL_SIZE // 2 # KERNEL_SIZE // 2 mean padding = 1 
    row_in = row_out - KERNEL_SIZE // 2

    # Allocate shared memory for one input channel
    sharedImg = cuda.shared.array(shape=(BLOCKSIZE, BLOCKSIZE), dtype=img.dtype)

    # Allocate shared memory for kernel of convolution
    sharedWeight = cuda.shared.array(shape=(KERNEL_SIZE, KERNEL_SIZE), dtype=weight.dtype)

    outPixel = bias[out_channel_idx]

    isOutputPixel = shared_col < OUTPUT_TILE_SIZE and shared_row < OUTPUT_TILE_SIZE and col_out < img.shape[3] and row_out < img.shape[2]

    for in_channel_idx in range(channel_in):
        #Copy kernel to shared memory
        if shared_col < KERNEL_SIZE and shared_row < KERNEL_SIZE:
            sharedWeight[shared_row][shared_col] = weight[out_channel_idx, in_channel_idx, shared_row, shared_col]

        #Copy img tile to shared memory
        if row_in >= 0 and row_in < img.shape[2] and col_in >= 0 and col_in < img.shape[3]:
            sharedImg[shared_row, shared_col] = img[batch_idx, in_channel_idx, row_in, col_in]
        else:
            sharedImg[shared_row, shared_col] = 0.0
        cuda.syncthreads()

        if isOutputPixel:
            for index in range(KERNEL_SIZE * KERNEL_SIZE):
                row = index // KERNEL_SIZE
                col = index % KERNEL_SIZE
                outPixel += sharedWeight[row, col] * sharedImg[shared_row + row, shared_col + col]

        cuda.syncthreads()

    if isOutputPixel:
        out_img[batch_idx, out_channel_idx, row_out, col_out] = outPixel


def Convolution2D_GPU(img, weight, bias):
    IMG_WIDTH, IMG_HEIGHT = img.shape[3], img.shape[2]
    BATCH_SIZE = img.shape[0]
    IN_CHANNEL, OUT_CHANNEL = weight.shape[1], weight.shape[0]

    threadPerBlock = (BLOCKSIZE, BLOCKSIZE)
    blockPerGrid = (math.ceil(IMG_WIDTH / OUTPUT_TILE_SIZE), math.ceil(IMG_HEIGHT / OUTPUT_TILE_SIZE), BATCH_SIZE * OUT_CHANNEL)

    d_img = cuda.to_device(img)
    d_out_img = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, IMG_HEIGHT, IMG_WIDTH), dtype= img.dtype)
    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)

    tileSharedConv2D_kernel[blockPerGrid, threadPerBlock](d_img, d_out_img, d_weight, d_bias, IN_CHANNEL, OUT_CHANNEL, BATCH_SIZE)

    out_img = d_out_img.copy_to_host()

    return out_img

@cuda.jit
def split_channel_tileSharedConv2D_kernel(img, out_img, weight, bias, channel_in: int, channel_out: int, batch_size: int, channel_per_thread: int):
    pass


def Split_Channel_TileSharedConv2D_GPU(img, out_img, weight, bias, channel_per_thread: int = 8):
    pass
    


@cuda.jit
def batchNorm2D_kernel(img, out_img, batchNorm_weight, batchNorm_bias, mean, variance,epsilon = 1e-5, batch_size = 1):
    out_x, out_y =cuda.grid(2)

    in_channel_idx = cuda.blockIdx.z % img.shape[1]

    batch_idx = cuda.blockIdx.z // img.shape[1]

    if out_x < img.shape[3] and out_y < img.shape[2] and cuda.blockIdx.z < img.shape[0]* img.shape[1]:
        norm_val = (img[batch_idx, in_channel_idx, out_y, out_x] - mean[in_channel_idx]) / math.sqrt(variance[in_channel_idx] +epsilon)
        out_img[batch_idx, in_channel_idx, out_y, out_x]


@cuda.jit
def ReLU(img, out_img):
    batch_idx = cuda.blockIdx.z // img.shape[1]
    in_channel_idx = cuda.blockIdx.z % img.shape[1]

    out_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    out_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if out_x < img.shape[3] and out_y < img.shape[2] and cuda.blockIdx.z < img.shape[0]* img.shape[1]:
        out_img[batch_idx, in_channel_idx, out_y, out_x] = math.max(0, img[batch_idx, in_channel_idx, out_y, out_x])


