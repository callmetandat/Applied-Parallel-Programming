from model_numba.Layers.init import *

@cuda.jit
def batchNorm2D_kernel(img, out_img, batchNorm_weight, batchNorm_bias, mean, variance,epsilon = 1e-5, batch_size = 1):
    out_x, out_y =cuda.grid(2)

    in_channel_idx = cuda.blockIdx.z % img.shape[1]

    batch_idx = cuda.blockIdx.z // img.shape[1]

    if out_x < img.shape[3] and out_y < img.shape[2] and cuda.blockIdx.z < img.shape[0]* img.shape[1]:
        norm_val = (img[batch_idx, in_channel_idx, out_y, out_x] - mean[in_channel_idx]) / math.sqrt(variance[in_channel_idx] +epsilon)
        out_img[batch_idx, in_channel_idx, out_y, out_x]

def batchNorm2D(img, batchNorm_weight, batchNorm_bias, mean, variance):
    pass