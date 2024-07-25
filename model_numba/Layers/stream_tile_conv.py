from model_numba.Layers.init import *

@cuda.jit
def Stream_tileSharedConv2D_kernel(img, out_img, weight, bias, channel_in: int, channel_out: int, batch_size: int, channel_per_thread: int):
    _, _, z_idx = cuda.grid(3)

    numBlockPerImg_Z_axis = (channel_in * channel_out) // channel_per_thread
    if z_idx >= batch_size * numBlockPerImg_Z_axis:
        return

    batch_idx = z_idx // numBlockPerImg_Z_axis
    out_channel_idx = (z_idx * channel_per_thread) % channel_out
    in_channel_offset = (z_idx % (channel_in // channel_per_thread)) * channel_per_thread

    shared_col, shared_row = cuda.threadIdx.x, cuda.threadIdx.y

    col_out = cuda.blockIdx.x * OUTPUT_TILE_SIZE + shared_col
    row_out = cuda.blockIdx.y * OUTPUT_TILE_SIZE + shared_row

    col_in = col_out - KERNEL_SIZE // 2  # KERNEL_SIZE // 2 means padding = 1
    row_in = row_out - KERNEL_SIZE // 2

    # Allocate shared memory for one input channel
    sharedImg = cuda.shared.array(shape=(INPUT_TILE_SIZE, INPUT_TILE_SIZE), dtype=numba.float32)


    outPixel = bias[out_channel_idx] if in_channel_offset == 0 else 0.0

    isOutputPixel = (shared_col < OUTPUT_TILE_SIZE and shared_row < OUTPUT_TILE_SIZE and 
                     col_out < img.shape[3] and row_out < img.shape[2])
    col, row = 0, 0
    for in_channel_idx in range(channel_per_thread):
        # Copy img tile to shared memory
        if row_in >= 0 and row_in < img.shape[2] and col_in >= 0 and col_in < img.shape[3]:
            sharedImg[shared_row, shared_col] = img[batch_idx, in_channel_offset + in_channel_idx, row_in, col_in]
        else:
            sharedImg[shared_row, shared_col] = 0.0
        
        cuda.syncthreads()

        if isOutputPixel:
            for index in range(KERNEL_SIZE_AS_1D_ARR):
                row = index // KERNEL_SIZE
                col = index % KERNEL_SIZE
                outPixel += weight[out_channel_idx, in_channel_idx + in_channel_offset , row, col] * sharedImg[shared_row + row, shared_col + col]

        cuda.syncthreads()

    if isOutputPixel:
        out_img[batch_idx, out_channel_idx, row_out, col_out] += outPixel


def Stream_TileSharedConv2D_GPU(img, weight, bias, channel_per_thread: int = 8):
    IMG_WIDTH, IMG_HEIGHT = img.shape[3], img.shape[2]
    BATCH_SIZE = img.shape[0]
    IN_CHANNEL, OUT_CHANNEL = weight.shape[1], weight.shape[0]

    threadPerBlock = (BLOCKSIZE, BLOCKSIZE)
    blockPerGrid = (math.ceil(IMG_WIDTH / OUTPUT_TILE_SIZE), math.ceil(IMG_HEIGHT / OUTPUT_TILE_SIZE), math.ceil(BATCH_SIZE * OUT_CHANNEL * (IN_CHANNEL // channel_per_thread)))

    out_img = cuda.pinned_array(shape=(BATCH_SIZE, OUT_CHANNEL, IMG_HEIGHT, IMG_WIDTH), dtype= img.dtype)
    with cuda.pinned(img, weight, bias):
        d_img = cuda.to_device(img)
        d_out_img = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, IMG_HEIGHT, IMG_WIDTH), dtype= img.dtype)
        d_weight = cuda.to_device(weight)
        d_bias = cuda.to_device(bias)

        Stream_tileSharedConv2D_kernel[blockPerGrid, threadPerBlock](d_img, d_out_img, d_weight, d_bias, IN_CHANNEL, OUT_CHANNEL, BATCH_SIZE, channel_per_thread)

        out_img = d_out_img.copy_to_host()

    return out_img