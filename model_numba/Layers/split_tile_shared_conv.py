from model_numba.Layers.init import *


@cuda.jit
def split_channel_tileSharedConv2D_kernel(img: Union[np.ndarray, any], out_img: Union[np.ndarray, any], weight: Union[np.ndarray, any], bias: Union[np.ndarray, any], channel_in: int, channel_out: int, batch_size: int, channel_per_thread: int):
    _, _, z_idx = cuda.grid(3)

    shared_col = cuda.threadIdx.x
    shared_row = cuda.threadIdx.y
    
    col_out = cuda.blockIdx.x * OUTPUT_TILE_SIZE + shared_col
    row_out = cuda.blockIdx.y * OUTPUT_TILE_SIZE + shared_row


    col_in = col_out - 1 # padding = 1
    row_in = row_out - 1 # padding = 1

    batch_idx = z_idx // (channel_out * (channel_in // channel_per_thread))
    out_channel_idx = z_idx % (channel_in // channel_per_thread)
    in_channel_offset = out_channel_idx * channel_per_thread

    shared_img = cuda.shared.array(shape=(INPUT_TILE_SIZE, INPUT_TILE_SIZE), dtype= img.dtype)

    isOutputPixel: bool = shared_col < OUTPUT_TILE_SIZE and shared_row < OUTPUT_TILE_SIZE and col_out < img.shape[3] and row_out < img.shape[2]

    outPixel: float = bias[out_channel_idx] if in_channel_offset == 0 else 0.0

    for c in range(channel_per_thread):
        #Copy data to Shared memory
        if row_in >= 0 and row_in < img.shape[2] and col_in >=0 and col_in < img.shape[3]:
            shared_img[shared_row, shared_col] = img[batch_idx, in_channel_offset + c, row_in, col_in]
        else:
            shared_img[shared_row, shared_col] = 0.0
        
        cuda.syncthreads()

        if isOutputPixel:
            for index in range(KERNEL_SIZE_AS_1D_ARR):
                row = index // KERNEL_SIZE
                col = index % KERNEL_SIZE
                outPixel += weight[out_channel_idx, in_channel_offset + c, row, col] * shared_img[shared_row + row, shared_col + col]

        cuda.syncthreads()

    if isOutputPixel:
        out_img[batch_idx, out_channel_idx, row_out, col_out] += outPixel



def Split_Channel_TileSharedConv2D_GPU(img, weight, bias, channel_per_thread: int = 8):
    IMG_WIDTH, IMG_HEIGHT = img.shape[3], img.shape[2]
    BATCH_SIZE = img.shape[0]
    IN_CHANNEL, OUT_CHANNEL = weight.shape[1], weight.shape[0]

    threadPerBlock = (BLOCKSIZE, BLOCKSIZE)
    blockPerGrid = (math.ceil(IMG_WIDTH / OUTPUT_TILE_SIZE), math.ceil(IMG_HEIGHT / OUTPUT_TILE_SIZE), math.ceil(BATCH_SIZE * OUT_CHANNEL * (IN_CHANNEL // channel_per_thread)))
  
    d_img = cuda.to_device(img)
    d_out_img = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, IMG_HEIGHT, IMG_WIDTH), dtype= img.dtype)
    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)

    split_channel_tileSharedConv2D_kernel[blockPerGrid, threadPerBlock](d_img, d_out_img, d_weight, d_bias, IN_CHANNEL, OUT_CHANNEL, BATCH_SIZE, channel_per_thread)

    out_img = d_out_img.copy_to_host()

    return out_img