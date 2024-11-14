import torch
import torch.nn.functional as F
import numpy as np

def add_noise_to_image(image, noise_level_min=0.1, noise_level_max=0.5, p_noise_mean=0, p_noise_std=1):
    """
    Adds Gaussian noise to an input image.

    Parameters:
    - image: Input image tensor of shape [3, 32, 32]
    - noise_level_min: Minimum noise level σ
    - noise_level_max: Maximum noise level σ
    - p_noise_mean: Mean of the Gaussian noise
    - p_noise_std: Standard deviation of the Gaussian noise

    Returns:
    - Noisy image
    """
    # Sample a random noise level σ
    sigma = np.random.uniform(noise_level_min, noise_level_max)

    # Generate Gaussian noise
    epsilon = np.random.normal(p_noise_mean, p_noise_std, image.shape)

    # Create the noisy image
    noisy_image = image + sigma * epsilon
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure pixel values are within [0, 1]

    return noisy_image, sigma


def preprocess_with_convolution(image, num_filters=16, kernel_size=3, stride=2, padding=1):
    """
    Applies a convolutional layer to an image to reduce its dimensionality.

    Args:
    - image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).
    - num_filters (int): Number of output channels (filters) for the convolution.
    - kernel_size (int): Size of the convolutional kernel.
    - stride (int): Stride for the convolution to control downsampling.
    - padding (int): Padding for the convolution to maintain spatial dimensions before downsampling.

    Returns:
    - torch.Tensor: Flattened tensor suitable for input to an MLP.
    """
    # Initialize convolutional filter weights and bias manually
    conv_weights = torch.randn(num_filters, image.size(1), kernel_size, kernel_size) * 0.1
    conv_bias = torch.randn(num_filters) * 0.1

    # Ensure the bias tensor has the same data type as the input image tensor
    conv_bias = conv_bias.to(image.dtype)

    # Apply convolution operation
    feature_map = F.conv2d(image, conv_weights, bias=conv_bias, stride=stride, padding=padding)
    
    # Flatten the output for MLP input
    flattened_output = feature_map.view(feature_map.size(0), -1)
    return flattened_output