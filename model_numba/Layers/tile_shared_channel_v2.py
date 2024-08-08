from model_numba.Layers.init import *

@cuda.jit
def Tile_conv_kernel(img: DeviceNDArray, out_img: DeviceNDArray, weight: DeviceNDArray, bias: DeviceNDArray, channel_in: int):

    x_idx, out_channel_idx, batch_idx = cuda.grid(3)

    shared_col, shared_row = cuda.threadIdx.x % INPUT_TILE_SIZE, cuda.threadIdx.x // INPUT_TILE_SIZE
   
    pass

    