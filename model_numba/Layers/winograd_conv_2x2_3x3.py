from model_numba.Layers.init import *


@cuda.jit(device= True, inline= True)
def filter_multiply(in_arr):
    out = [0.0] * 4
    out[0] = in_arr[0]
    temp = in_arr[0] + in_arr [2]
    out[1] = 0.5 * (temp + in_arr[1])
    out[2] = 0.5 * (temp - in_arr[1])
    out[3] = out[2]
    return out


@cuda.jit
def wino_filter_transform(g: np.ndarray, filter_transform: any, in_channel: int, out_channel: int):

    NUM_KERNEL = in_channel * out_channel
    channel_idx_offset = cuda.blockIdx.x * NUM_KERNELS_PER_BLOCK

    shared_g = cuda.shared.array(shape=(NUM_KERNELS_PER_BLOCK, 4, 4),dtype=g.dtype)

    #Load data to Shared Memory
    for i in range(cuda.threadIdx.x, NUM_KERNELS_PER_BLOCK * KERNEL_SIZE_AS_1D_ARR, BLOCK_SIZE_FC): #KERNEL_SIZE_AS_1D_ARR = 9, NUM_KERNELS_PER_BLOCK =40
        col_idx = i % KERNEL_SIZE #KERNEL_SIZE =3
        row_idx = (i // KERNEL_SIZE) % KERNEL_SIZE
        channel_idx = (i // KERNEL_SIZE) // KERNEL_SIZE
        global_channel_idx = channel_idx + channel_idx_offset

        if global_channel_idx < NUM_KERNEL:
            shared_g[channel_idx, row_idx, col_idx] = g[global_channel_idx * KERNEL_SIZE_AS_1D_ARR + row_idx * KERNEL_SIZE + col_idx]

    
    cuda.syncthreads()

    shared_Gg = cuda.shared.array(shape=(NUM_KERNELS_PER_BLOCK, 4, 3), dtype= g.dtype)

    
    for i in range(cuda.threadIdx.x, NUM_KERNEL * KERNEL_SIZE, BLOCK_SIZE_FC):
        col_idx = i % KERNEL_SIZE
        channel_idx = i // KERNEL_SIZE

        if channel_idx + channel_idx_offset < NUM_KERNEL:
            in_col = [shared_g[channel_idx, row_idx, col_idx] for row_idx in range(KERNEL_SIZE)]
            

            #Compute Out_COL
            out_col = filter_multiply(in_col)

            for row_idx in range(4):
                shared_Gg [channel_idx, row_idx, col_idx] = out_col[row_idx]

    cuda.syncthreads()
 
    for i in range(cuda.threadIdx.x, NUM_KERNELS_PER_BLOCK *4, BLOCK_SIZE_FC):
        row_idx = i % 4
        channel_idx = i // 4

        if (channel_idx + channel_idx_offset < NUM_KERNEL):
            in_row = [shared_Gg[channel_idx, row_idx,col_idx] for col_idx in range(3)]

            out_row = filter_multiply(out_row)

            for row_idx in range(4):
                shared_g[channel_idx, row_idx, col_idx] = out_row[col_idx]

    cuda.syncthreads()

    for i in range(cuda.threadIdx.x, 4*4* NUM_KERNELS_PER_BLOCK, BLOCK_SIZE_FC):
        channel_idx = i % NUM_KERNELS_PER_BLOCK
        offset = i // NUM_KERNELS_PER_BLOCK

        col_idx = offset % 4
        row_idx = offset //4

        global_channel_idx = channel_idx_offset + channel_idx
        if (global_channel_idx < NUM_KERNEL):
            filter_transform[offset * NUM_KERNEL + global_channel_idx] = shared_g[channel_idx, row_idx, col_idx]
            










    


    
