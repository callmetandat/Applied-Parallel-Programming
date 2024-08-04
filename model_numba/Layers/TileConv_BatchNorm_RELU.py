from model_numba.Layers.init import *


@cuda.jit
def combine_TileConv_batchNorm_RELU(img: DeviceNDArray, out_img: DeviceNDArray, weight: DeviceNDArray, bias,
                                    norm_weight, norm_bias, mean, var,
                                    channel_in: int, channel_out: int, batch_size: int, epsilon=1e-5):

    _, _, z_idx = cuda.grid(3)

    batch_idx, out_channel_idx = z_idx // channel_out, z_idx % channel_out
    shared_col, shared_row = cuda.threadIdx.x, cuda.threadIdx.y

    col_out = cuda.blockIdx.x * OUTPUT_TILE_SIZE + shared_col
    row_out = cuda.blockIdx.y * OUTPUT_TILE_SIZE + shared_row

    col_in = col_out - KERNEL_SIZE // 2  # KERNEL_SIZE // 2 means padding = 1
    row_in = row_out - KERNEL_SIZE // 2

    #Copy data to constant memory for batchNorm step
    d_norm_weight = cuda.const.array_like(norm_weight)
    d_norm_bias = cuda.const.array_like(norm_bias)
    d_norm_mean = cuda.const.array_like(mean)
    d_norm_var = cuda.const.array_like(var)

    # Allocate shared memory for one input channel
    sharedImg = cuda.shared.array(shape=(INPUT_TILE_SIZE, INPUT_TILE_SIZE), dtype=img.dtype)

    outPixel = bias[out_channel_idx]

    isOutputPixel = shared_col < OUTPUT_TILE_SIZE and shared_row < OUTPUT_TILE_SIZE and col_out < img.shape[3] and row_out < img.shape[2]

    for in_channel_idx in range(channel_in):
        # Copy img tile to shared memory
        if row_in >= 0 and row_in < img.shape[2] and col_in >= 0 and col_in < img.shape[3]:
            sharedImg[shared_row, shared_col] = img[batch_idx, in_channel_idx, row_in, col_in]
        else:
            sharedImg[shared_row, shared_col] = 0.0
        cuda.syncthreads()

        if isOutputPixel:
            for index in range(KERNEL_SIZE_AS_1D_ARR):
                row = index // KERNEL_SIZE
                col = index % KERNEL_SIZE
                if (shared_row + row < INPUT_TILE_SIZE) and (shared_col + col < INPUT_TILE_SIZE):
                    outPixel += weight[out_channel_idx, in_channel_idx, row, col] * sharedImg[shared_row + row, shared_col + col]

        cuda.syncthreads()

    if isOutputPixel:
        outPixel = d_norm_weight[out_channel_idx] * (outPixel - d_norm_mean[out_channel_idx]) / math.sqrt(d_norm_var[out_channel_idx] + epsilon) + d_norm_bias[out_channel_idx]
        outPixel = max(0, outPixel)
        out_img[batch_idx, out_channel_idx, row_out, col_out] = outPixel


def Combie_TileConv_GPU(img: Union[np.ndarray, DeviceNDArray], weight: np.ndarray, bias: np.ndarray,
                        norm_weight: np.ndarray, norm_bias: np.ndarray, mean: np.ndarray, var: np.ndarray,
                        convert_output_to_host: bool = True) -> Union[np.ndarray, DeviceNDArray]:

    IMG_WIDTH, IMG_HEIGHT = img.shape[3], img.shape[2]
    BATCH_SIZE = img.shape[0]
    IN_CHANNEL, OUT_CHANNEL = weight.shape[1], weight.shape[0]

    threadPerBlock = (BLOCKSIZE, BLOCKSIZE)
    blockPerGrid = (math.ceil(IMG_WIDTH / OUTPUT_TILE_SIZE), math.ceil(IMG_HEIGHT / OUTPUT_TILE_SIZE), BATCH_SIZE * OUT_CHANNEL)

    # Allocate and transfer data
    d_img = cuda.to_device(img) if isinstance(img, np.ndarray) else img
    d_out_img = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, IMG_HEIGHT, IMG_WIDTH), dtype=img.dtype)
    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)

    norm_weight = cuda.to_device(norm_weight)
    norm_bias = cuda.to_device(norm_bias)
    mean= cuda.to_device(mean)
    var = cuda.to_device(var)

    # Run Kernel
    combine_TileConv_batchNorm_RELU[blockPerGrid, threadPerBlock](d_img, d_out_img, d_weight, d_bias, norm_weight, norm_bias, mean, var, IN_CHANNEL, OUT_CHANNEL, BATCH_SIZE, 1e-5)

    if convert_output_to_host:
        out_img = d_out_img.copy_to_host()
    else:
        out_img = d_out_img

    return out_img