import numpy as np

class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, weights=None, bias=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases if not provided
        self.weights = weights if weights is not None else np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = bias if bias is not None else np.zeros((out_channels,))

    def forward(self, input_data):
        batch_size, in_channels, input_height, input_width = input_data.shape
        output_height = int((input_height + 2 * self.padding - self.kernel_size) / self.stride) + 1
        output_width = int((input_width + 2 * self.padding - self.kernel_size) / self.stride) + 1
        output = np.zeros((batch_size, self.out_channels, output_height, output_width))

        # Pad the input if necessary
        if self.padding > 0:
            input_data = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for b in range(batch_size):
            for c in range(self.out_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        output[b, c, h, w] = np.sum(input_data[b, :, h_start:h_end, w_start:w_end] * self.weights[c]) + self.bias[c]

        return output
