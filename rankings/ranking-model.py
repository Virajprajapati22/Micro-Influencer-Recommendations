import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
from keras.initializers import RandomUniform
import numpy as np

no_of_features = 2
no_of_samples = 4000
eps = 1e-10

# def listNet(y_pred, y_true):
#     """
#     ListNet loss introduced in "Seeking Micro-influencers for Brand Promotion".

#     :param y_pred: predictions from the model, shape [batch_size, slate_length]
#     :param y_true: ground truth labels, shape [batch_size, slate_length]
#     :param eps: epsilon value, used for numerical stability
#     :nBrands: total number of brands in dataset
#     :nSamples: total number of micro-influencers samples in particular brand
#     :lamda: regularization rate
#     :return: loss value
#     """

#     # Compute softmax probabilities over the predicted ranking scores
#     softmax_pred = tf.nn.softmax(y_pred, axis=-1)

#     # Compute softmax probabilities over the actual ranking scores
#     softmax_pred = tf.nn.softmax(y_true, axis=-1)

#     # Compute cross-entropy loss between true rankings and predicted softmax probabilities
#     loss = -tf.reduce_mean(tf.reduce_sum(y_true * tf.math.log(softmax_pred + eps), axis=-1))
    
#     return loss

def create_listnet_model(input_shape, num_nodes):
    """
    Create a ListNet neural network model using keras.

    Parameters:
        input_shape: Tuple. Shape of the input features.
        num_nodes: Int. Number of nodes in the hidden layers.

    Returns:
        model: ListNet neural network model.
    """

    model = Sequential([
        Dense(num_nodes, activation="leaky_relu", input_shape=input_shape, kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1)),
        Dense(num_nodes, activation="leaky_relu", kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1)),
        Dense(num_nodes, activation="leaky_relu", kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1)),
        Dense(1, activation="softmax", kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1)) # output layer
    ])

    return model
