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
    
class BatchNorm2D_CPU:
    def __init__(self, num_channels, weight, bias, epsilon=1e-5):
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.gamma = weight
        self.beta = bias

    def forward(self, x, mean, variance):
        """
        Apply batch normalization to a 4D input tensor.
        
        Parameters:
        - x: Input tensor of shape (batch_size, num_channels, height, width)
        - mean: Mean per channel
        - variance: Variance per channel
        
        Returns:
        - Normalized tensor of the same shape as input
        """
 
        assert mean.shape == (self.num_channels,)
        assert variance.shape == (self.num_channels,)
        

        batch_size, num_channels, height, width = x.shape
        y = np.empty_like(x)
        
        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(height):
                    for j in range(width):
                        y[b, c, i, j] = (
                            self.gamma[c] * (x[b, c, i, j] - mean[c]) / np.sqrt(variance[c] + self.epsilon)
                            + self.beta[c]
                        )
        
        return y



class MaxPooling2D_CPU:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        """
        Apply MaxPooling to a 4D input tensor.
        
        Parameters:
        - x: Input tensor of shape (batch_size, num_channels, height, width)
        
        Returns:
        - Pooled tensor of shape (batch_size, num_channels, pooled_height, pooled_width)
        """
        batch_size, num_channels, height, width = x.shape
        pooled_height = (height - self.pool_size) // self.stride + 1
        pooled_width = (width - self.pool_size) // self.stride + 1
        
        y = np.zeros((batch_size, num_channels, pooled_height, pooled_width), dtype=x.dtype)

        for b in range(batch_size):
            for c in range(num_channels):
                for i in range(pooled_height):
                    for j in range(pooled_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size

                        # Manually calculate the max value in the pooling region
                        max_val = float('-inf')
                        for h in range(h_start, h_end):
                            for w in range(w_start, w_end):
                                if x[b, c, h, w] > max_val:
                                    max_val = x[b, c, h, w]
                        y[b, c, i, j] = max_val

        return y
    

class ReLU_CPU:
    def __init__(self):
        pass

    def forward(self, x):
        """
        Apply ReLU activation to a 4D input tensor.
        
        Parameters:
        - x: Input tensor of shape (batch_size, num_channels, height, width)
        
        Returns:
        - Activated tensor of the same shape as input
        """
        batch_size, num_channels, height, width = x.shape
        y = np.zeros_like(x)

        for b in range(batch_size):
            for c in range(num_channels):
                for h in range(height):
                    for w in range(width):
                        y[b, c, h, w] = max(x[b, c, h, w], 0)

        return y