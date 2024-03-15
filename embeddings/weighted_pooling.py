import numpy as np

def weighted_history_pooling(features, gamma):
    """
    Apply weighted history pooling to the input features.

    Parameters:
    - features: Input features (N x M).
    - gamma: Scaling factor for the exponential function.

    Returns:
    - Weighted pooled features (1 x M).
    """
    # Calculate mean and standard deviation along the rows (axis=0)
    mean = np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)
    
    # Calculate weights based on the standard deviation
    weights = np.zeros_like(mean)  # Initialize weights array with zeros
    non_zero_mean_indices = mean != 0
    weights[non_zero_mean_indices] = np.exp(-gamma * (std_dev[non_zero_mean_indices] / mean[non_zero_mean_indices]))
    
    # Calculate weighted average pooling
    weighted_pooling = np.mean(features, axis=0) * weights # element wise multiplication
    
    return weighted_pooling
