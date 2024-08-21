from model_numba.Layers.init import *


@cuda.jit
def directConv2D_kernel(img, output, weight, bias, in_channel, out_channel, batch_size):
    out_row, out_col, z_idx = cuda.grid(3)

    batch_idx = z_idx // out_channel
    out_channel_idx = z_idx % out_channel

    if out_row < output.shape[2] and out_col < output.shape[3]:
        outPixel = bias[out_channel_idx]
        
        for in_channel_index in range(in_channel):
            for k_row in range(KERNEL_SIZE):
                for k_col in range(KERNEL_SIZE):
                    outPixel += weight[out_channel_idx, in_channel_index, k_row, k_col] *\
                          img[batch_idx, in_channel_index, out_row + k_row, out_col + k_col]
                    
        output[batch_idx, out_channel_idx, out_row, out_col] = outPixel


def DirectConv2DGPU(img, weight, bias, convert_to_host=False):
    import time
    time.sleep(0.4)
    BATCH_SIZE, IN_CHANNEL, HEIGHT, WIDTH = img.shape
    OUT_CHANNEL = weight.shape[0]
    PADDING = 1

    img = np.pad(img, ((0, 0), (0, 0), (PADDING, PADDING), (PADDING, PADDING)), mode='constant', constant_values=0.0)

    with cuda.pinned(weight, bias):
        d_img = cuda.to_device(img)
        d_weight = cuda.to_device(weight)
        d_bias = cuda.to_device(bias)

        d_out = cuda.device_array(shape=(BATCH_SIZE, OUT_CHANNEL, HEIGHT, WIDTH), dtype=img.dtype)
        
        threadPerBlock = (BLOCKSIZE, BLOCKSIZE)
        blockPerGrid = (math.ceil(WIDTH / BLOCKSIZE), math.ceil(HEIGHT / BLOCKSIZE), BATCH_SIZE * OUT_CHANNEL)
        
        directConv2D_kernel[blockPerGrid, threadPerBlock](d_img, d_out, d_weight, d_bias, IN_CHANNEL, OUT_CHANNEL, BATCH_SIZE)

        output = d_out.copy_to_host() if convert_to_host else d_out

        return output










