from model_numba.Layers.init import *

@cuda.jit
def Conv_1D_Filter_kernel(img: Union[np.ndarray, DeviceNDArray], out_img: Union[np.ndarray, DeviceNDArray],
                          weight: DeviceNDArray, bias: DeviceNDArray, in_channel: int):
    x_idx, batch_idx = cuda.grid(2)
    if x_idx < img.shape[2] * img.shape[3]:
        row, col = x_idx // img.shape[3], x_idx % img.shape[3]

        outPixel = bias[0]
        for i in range(in_channel):
            outPixel += weight[0, i, 0, 0] * img[batch_idx, i, row, col]

        out_img[batch_idx, 0, row, col] = outPixel

def Conv_1D_Filter_GPU(img: Union[np.ndarray, DeviceNDArray], weight: np.ndarray, bias: np.ndarray):
    BATCH_SIZE, IN_CHANNEL, HEIGHT, WIDTH = img.shape
    OUT_CHANNEL = 1

    d_weight = cuda.to_device(weight)
    d_bias = cuda.to_device(bias)
    d_img = cuda.to_device(img) if isinstance(img, np.ndarray) else img

    d_output = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, HEIGHT, WIDTH), dtype=img.dtype)

    threadPerBlock = (BLOCK_SIZE_FC,)
    blockPerGrid = (math.ceil(HEIGHT * WIDTH / BLOCK_SIZE_FC), BATCH_SIZE)

    Conv_1D_Filter_kernel[blockPerGrid, threadPerBlock](d_img, d_output, d_weight, d_bias, IN_CHANNEL)

    output = d_output.copy_to_host()
    return output