import numpy as np
import csv

def initialize_parameters(dv, dt, dh1, dh2, da):
    """
    Initialize parameters for the Multimodal Social Account Embedding.

    Parameters:
    - dv: Dimensionality of visual features.
    - dt: Dimensionality of textual features.
    - dh1: Dimensionality of the first hidden state vector.
    - dh2: Dimensionality of the second hidden state vector.
    - da: Dimensionality of the final social account representation.

    Returns:
    - Parameters Wt1, bt1, Wt2, Wv1, bv1, Wv2.
    """
    # Initialize parameters with uniform distribution between 0.1 and -0.1
    Wt1 = np.random.uniform(-0.1, 0.1, size=(dt, dh1))
    bt1 = np.random.uniform(-0.1, 0.1, size=(1, dh1))
    Wt2 = np.random.uniform(-0.1, 0.1, size=(dh1, da))
    
    Wv1 = np.random.uniform(-0.1, 0.1, size=(dv, dh2))
    bv1 = np.random.uniform(-0.1, 0.1, size=(1, dh2))
    Wv2 = np.random.uniform(-0.1, 0.1, size=(dh2, da))
    
    return Wt1, bt1, Wt2, Wv1, bv1, Wv2

def non_linear_activation(x, alpha):
    """
    Apply Leaky ReLU activation function to the input.

    Parameters:
    - x: Input vector.
    - alpha: Slope for negative values (default is 0.01 for slight leakiness).

    Returns:
    - Output after applying the activation function.
    """
    return np.maximum(alpha * x, x)

def multimodal_social_account_embedding(ev, et, Wt1, bt1, Wt2, Wv1, bv1, Wv2, alpha):
    """
    Perform Multimodal Social Account Embedding.

    Parameters:
    - ev: Visual features.
    - et: Textual features.
    - Wt1, bt1, Wt2, Wv1, bv1, Wv2: Parameters.

    Returns:
    - Final social account representation.
    """
    # Linear transformation followed by non-linear activation for textual features
    ht1 = non_linear_activation(np.dot(et, Wt1) + bt1, alpha)
    ht2 = np.dot(ht1, Wt2)
    
    # Linear transformation followed by non-linear activation for visual features
    hv1 = non_linear_activation(np.dot(ev, Wv1) + bv1, alpha)
    hv2 = np.dot(hv1, Wv2)
    
    # Concatenate and compute inner product to obtain final social account representation
    ea = np.dot(ht2, hv2.T)
    
    return ea

# Example usage
dv = 25088  # Dimensionality of visual features
dt = 300   # Dimensionality of textual features
dh1 = 300  # Dimensionality of the first hidden state vector
dh2 = 4096  # Dimensionality of the second hidden state vector
da = 512   # Dimensionality of the final social account representation
alpha = 0.5

# Initialize parameters
Wt1, bt1, Wt2, Wv1, bv1, Wv2 = initialize_parameters(dv, dt, dh1, dh2, da)

# Initialize empty lists to store visual and textual features
ev = []
et = []

# Read visual features from CSV
with open('embeddings/extracted_features_files/visual_features.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        ev.append([float(val) for val in row])

# Read textual features from CSV
with open('embeddings/extracted_features_files/textual_features.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        et.append([float(val) for val in row])

# Perform Multimodal Social Account Embedding
ea = multimodal_social_account_embedding(ev, et, Wt1, bt1, Wt2, Wv1, bv1, Wv2, alpha)

print("Final social account representation:")
print(ea)
