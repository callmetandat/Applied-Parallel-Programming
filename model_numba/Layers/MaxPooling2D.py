from model_numba.Layers.init import *



@cuda.jit
def MaxPooling_Kernel(input: DeviceNDArray, output: DeviceNDArray, num_channels: int, output_height: int, output_width: int):
    index = cuda.grid(1)
    total_output_elements = output.size

    if index < total_output_elements:
        batch_idx = index // (num_channels * output_height * output_width)
        channel_idx = (index // (output_height * output_width)) % num_channels
        row = (index // output_width) % output_height
        col = index % output_width
        
        input_row = row * STRIDE
        input_col = col * STRIDE
        
        max_val = input[batch_idx, channel_idx, input_row, input_col]
        max_val = max(max_val, input[batch_idx, channel_idx, input_row, input_col + 1])
        max_val = max(max_val, input[batch_idx, channel_idx, input_row + 1, input_col])
        max_val = max(max_val, input[batch_idx, channel_idx, input_row + 1, input_col + 1])
        
        output[batch_idx, channel_idx, row, col] = max_val

def MaxPooling2D_GPU(input: Union[np.ndarray | DeviceNDArray], convert_to_host: bool = False) -> Union[np.ndarray, DeviceNDArray]:
    batch_size, channel, input_height, input_width = input.shape

    output_height = (input_height + STRIDE - 1) // STRIDE
    output_width = (input_width + STRIDE - 1) // STRIDE
    
    total_output_elements = batch_size * channel * output_height * output_width

    thread_per_block = 256
    block_per_grid = (math.ceil(total_output_elements / thread_per_block), )

    d_input = cuda.to_device(input) if isinstance(input, np.ndarray) else input
    d_output = cuda.device_array(shape=(batch_size, channel, output_height, output_width), dtype=input.dtype)

    MaxPooling_Kernel[block_per_grid, thread_per_block](d_input, d_output, channel, output_height, output_width)

    output = d_output.copy_to_host() if convert_to_host else d_output

    return output