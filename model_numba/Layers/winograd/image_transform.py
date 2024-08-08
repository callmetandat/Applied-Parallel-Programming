from model_numba.Layers.init import *

@cuda.jit(device=True, inline=True)
def multiply_input(input):
    output = [0.0] * 3

    output[0] = input[0] - input[2]
    output[1] = input[1] + input[2]
    output[2] = input[2] - input[1]
    output[3] = input[1] - input[3]
    return output


@cuda.jit
def input_transform_kernel(input: DeviceNDArray, input_transform: DeviceNDArray, batch_size: int, channel: int,\
                           H: int, W: int, tile_H: int, tile_W: int):
    tile_col_idx_start = cuda.blockIdx.x * TILES_X_PER_BLOCK