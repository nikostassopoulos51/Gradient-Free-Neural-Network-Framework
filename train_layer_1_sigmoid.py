'''
Trains the first layer of a neural network using the sigmoid activation function.

Parameters:
    neurons (int): Number of neurons in the layer.
    vars (int): Number of input variables.
    obs (int): Number of observations.
    n_per_part (list): Indices indicating the partitioning of the dataset.
    inds_all (list): Indices for all observations.
    x (numpy.ndarray): Input features.
    y (numpy.ndarray): Target outputs.

Returns:
    tuple: A tuple containing:
        - a_all (list): Weights for each neuron.
        - a_layer1 (numpy.ndarray): Weights for the output layer.
        - layer1 (numpy.ndarray): Activated output of the first layer.
'''

import numpy as np
from datetime import datetime

def sigm1(x):
    return 1 / (1 + np.exp(-x))  # Sigmoid function

def isigm1(y):
    return -np.log(1 / y - 1)  # Inverse sigmoid (for valid y values in (0,1))

def train_layer_1_sigmoid(neurons, vars, obs, n_per_part, inds_all, x, y):
    
    a_all = []
    ooops = 0

    for i in range(neurons):

        try:

            if i + 1 >= len(n_per_part):
                break  # Prevent accessing out-of-bounds indices

            current_indices = inds_all[n_per_part[i]:n_per_part[i + 1]]
            x_slice = x[current_indices]
            y_slice = y[current_indices]

            atmp = np.linalg.lstsq(np.hstack((np.ones((x_slice.shape[0], 1)), x_slice)), isigm1(y_slice), rcond=None)[0]

            if np.sum(np.isnan(atmp)) == 0 and np.sum(np.isinf(atmp)) == 0:
                a_all.append(atmp)  # Append valid weights

            else:
                ooops += 1
                a_all.append(np.zeros(vars + 1))  # Append zero weights if invalid

        except Exception as ex:
            ooops += 1
            print(f"Problem Neuron = {i}, Total Ooops = {ooops}, Ex = {ex}")
            a_all.append(np.zeros(vars + 1))
    
    layer1 = np.zeros((obs, neurons))

    for i in range(neurons):
        if i % 100 == 0:  # Check if i is a multiple of 100
            current_time = datetime.now().strftime("%H:%M:%S")  # Get the current time in HH:MM:SS format
            print(f"{current_time} Applying sigmoid on {i} neuron, of {neurons} Total")

        layer1[:, i] = sigm1(np.dot(np.hstack((np.ones((obs, 1)), x)), a_all[i]))  # type: ignore #sigm1 is included in the ANNBN file (the main one) # Perform the operation
    
    a_layer1 = np.linalg.lstsq(np.hstack((layer1, np.ones((obs, 1)))), y, rcond = None)[0]
    
    return a_all, a_layer1, layer1

# Test Parameters
neurons = 5  # Number of neurons
vars = 3  # Number of input variables
obs = 20  # Number of observations
n_per_part = [0, 5, 10, 15, 20]  # Partition indices
inds_all = list(range(obs))  # All indices

# Generate synthetic data
np.random.seed(42)  # For reproducibility
x = np.random.rand(obs, vars)  # Random input features
y = sigm1(np.random.rand(obs))  # Random sigmoid-transformed target outputs

# Call the function
a_all, a_layer1, layer1 = train_layer_1_sigmoid(neurons, vars, obs, n_per_part, inds_all, x, y)

# Print the outputs for validation
print("a_all (Weights for each neuron):")
print(a_all)

print("\na_layer1 (Output layer weights):")
print(a_layer1)

print("\nlayer1 (Activated output of the first layer):")
print(layer1)