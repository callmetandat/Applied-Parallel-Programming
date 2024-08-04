from numba import cuda
import numba
import numpy as np
import math
from typing import Union
from numba.cuda.cudadrv.devicearray import DeviceNDArray

BLOCKSIZE = 16
KERNEL_SIZE = 3
KERNEL_SIZE_AS_1D_ARR = KERNEL_SIZE * KERNEL_SIZE
OUTPUT_TILE_SIZE = BLOCKSIZE - KERNEL_SIZE + 1
INPUT_TILE_SIZE = BLOCKSIZE
STRIDE = 2

# Config for winograd comvolution_2x2_3x3
BLOCK_SIZE_FC = 128
NUM_KERNELS_PER_BLOCK = 40