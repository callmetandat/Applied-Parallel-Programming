from model_numba.Layers.init import *


@cuda.jit
def TransposeConv2D_kernel(img: DeviceNDArray, out_img: DeviceNDArray, weight: DeviceNDArray, bias: DeviceNDArray, channel_in: int, channel_out: int, batch_size: int):
    x_idx, out_channel_idx, batch_idx = cuda.grid(3)
    
    total_elements = img.shape[2] * img.shape[3]

    if x_idx < total_elements and out_channel_idx < channel_out and batch_idx < batch_size:
        row_out, col_out = x_idx // img.shape[3], x_idx % img.shape[3]
        row_in, col_in = row_out // TRANSPOSE_STRIDE, col_out // TRANSPOSE_STRIDE

        outPixel = bias[out_channel_idx]

        for in_channel_idx in range(channel_in):
            k_row, k_col = row_out % TRANSPOSE_STRIDE, col_out % TRANSPOSE_STRIDE

            outPixel += weight[in_channel_idx, out_channel_idx, k_row, k_col] * img[batch_idx, in_channel_idx, row_in, col_in]

        out_img[batch_idx, out_channel_idx, row_out, col_out] = outPixel

def TransposeConvol2D_GPU(img: Union[np.ndarray, DeviceNDArray], weight: np.ndarray, bias: np.ndarray, convert_to_host: bool = False):
    
    with cuda.pinned(weight, bias):
        d_img = cuda.to_device(img) if isinstance(img, np.ndarray) else img
        d_weight = cuda.to_device(weight)
        d_bias = cuda.to_device(bias)

        CHANNEL_IN, CHANNEL_OUT = weight.shape[0], weight.shape[1]
        BATCH_SIZE = img.shape[0]
        _, _, HEIGHT, WIDTH = img.shape
        total_elements = HEIGHT * WIDTH

        threadPerBlock = (BLOCK_SIZE_FC, 1, 1)
        blockPerGrid = (math.ceil(total_elements / BLOCK_SIZE_FC), CHANNEL_OUT, BATCH_SIZE)

        d_out_img = cuda.device_array(shape=(BATCH_SIZE, CHANNEL_OUT, HEIGHT * 2, WIDTH * 2), dtype=img.dtype)

        # Run Kernel
        TransposeConv2D_kernel[blockPerGrid, threadPerBlock](d_img, d_out_img, d_weight, d_bias, CHANNEL_IN, CHANNEL_OUT, BATCH_SIZE)

        out_img = d_out_img.copy_to_host() if convert_to_host else d_out_img

    return out_img