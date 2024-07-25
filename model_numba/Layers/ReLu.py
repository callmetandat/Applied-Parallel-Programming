from model_numba.Layers.init import *

@cuda.jit
def ReLU(img, out_img):
    batch_idx = cuda.blockIdx.z // img.shape[1]
    in_channel_idx = cuda.blockIdx.z % img.shape[1]

    # out_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # out_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    out_x, out_y = cuda.grid(2)

    if out_x < img.shape[3] and out_y < img.shape[2] and cuda.blockIdx.z < img.shape[0]* img.shape[1]:
        out_img[batch_idx, in_channel_idx, out_y, out_x] = math.max(0, img[batch_idx, in_channel_idx, out_y, out_x])

