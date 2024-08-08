from model_numba.Layers.init import *


@cuda.jit(device=True, inline= True)
def multiply_filter(input):
    output = [0.0] * 3
    output[0] = input[0]
    temp = input[0] + input[2]
    output[1] = 0.5 * (temp + input[1])
    output[2] = 0.5 * (temp - input[1])
    output[3] = input[2]

    return output

@cuda.jit
def filter_transform_kernel(in_filter: DeviceNDArray, out_filter: DeviceNDArray, channel_in: int, channel_out: int):
    NUM_KERNEL = channel_in * channel_out
    channel_idx_start = cuda.blockIdx.x * NUM_KERNELS_PER_BLOCK

    shared_g = cuda.shared.array(shape=(NUM_KERNELS_PER_BLOCK, 4, 4), dtype= in_filter.dtype)


    # Load data to shared memory
    for i in range(cuda.threadIdx.x, NUM_KERNELS_PER_BLOCK *3 *3, BLOCK_SIZE_FC):
        col_idx = i % 3
        row_idx = (i //3) % 3
        channel_idx = (i //3) // 3

        global_channel_idx = channel_idx + channel_idx_start
        if global_channel_idx < NUM_KERNEL:
            shared_g[channel_idx, row_idx, col_idx] = in_filter[global_channel_idx * 9 + row_idx*3 + col_idx]

    cuda.syncthreads()

    # Compute Gg
    shared_Gg = cuda.shared.array(shape=(NUM_KERNELS_PER_BLOCK, 4, 3), dtype= in_filter.dtype)

    for i in range(cuda.threadIdx.x , NUM_KERNELS_PER_BLOCK*3, BLOCK_SIZE_FC):
        col_idx, channel_idx = i % 3, i//3

        if channel_idx_start + channel_idx < NUM_KERNEL:
            in_col = [shared_g[channel_idx, row_idx, col_idx] for row_idx in range(3)]

            out_col = multiply_filter(in_col)

            for row_idx in range(4):
                shared_Gg[channel_idx, row_idx, col_idx] = out_col [row_idx]

    cuda.syncthreads()

    # compute GgGT

    for i in range(cuda.threadIdx.x, NUM_KERNELS_PER_BLOCK * 4, NUM_KERNELS_PER_BLOCK):
        row_idx, channel_idx = i % 4, i //4

        if channel_idx + channel_idx_start < NUM_KERNEL:
            in_row = [shared_Gg[channel_idx, row_idx, col_idx] for col_idx in range(3)]

            out_row = multiply_filter(in_row)

            for col_idx in range(4):
                shared_g[channel_idx, row_idx, col_idx] = out_row[col_idx]

    cuda.syncthreads()

    # Loading data for GMEM

    for i in range(cuda.threadIdx.x, NUM_KERNELS_PER_BLOCK *4 *4, BLOCK_SIZE_FC):
        channel_idx = i % NUM_KERNELS_PER_BLOCK
        offset = i // NUM_KERNELS_PER_BLOCK

        col_idx, row_idx = offset % 4, offset // 4

        global_channel_idx = channel_idx + channel_idx_start
        if global_channel_idx < NUM_KERNEL:
            out_filter[offset * NUM_KERNEL + global_channel_idx] = shared_g[channel_idx, row_idx, col_idx]