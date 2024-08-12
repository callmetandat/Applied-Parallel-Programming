from model_numba.Layers.init import *

@cuda.jit
def relu_kernel(data, batch_size, channels, img_width, img_height):
    idx = cuda.grid(1)
    total_elements = batch_size * channels * img_width * img_height

    if idx < total_elements:
        b = idx // (channels * img_width * img_height)
        c = (idx % (channels * img_width * img_height)) // (img_width * img_height)
        w = (idx % (img_width * img_height)) // img_height
        h = idx % img_height
        data[b, c, w, h] = max(data[b, c, w, h], 0)


def RELU_GPU(data):
    d_data = cuda.to_device(data)

    batch_size, channels, img_width, img_height = data.shape
    total_elements = batch_size * channels * img_width * img_height
    threads_per_block = 256
    blocks_per_grid = (total_elements + threads_per_block - 1) // threads_per_block
    
    relu_kernel[blocks_per_grid, threads_per_block](d_data, batch_size, channels, img_width, img_height)
    out_put = d_data.copy_to_host()
    return out_put



