import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
from datetime import datetime
import time

# KMeans clustering function (used for splitting the data into clusters)
def ___clustering(neurons, xx_clust, max_iter):
    kmeans = KMeans(n_clusters=neurons, max_iter=max_iter, n_init=10, random_state=51)
    kmeans.fit(xx_clust)
    
    assignments = kmeans.labels_
    counts = np.bincount(assignments)
    
    sorted_indices = np.argsort(assignments)
    inds_all = sorted_indices.tolist()
    items_per_neuron = counts.tolist()
    n_per_part = [0] + np.cumsum(items_per_neuron).tolist()

    return inds_all, n_per_part, items_per_neuron

# Sigmoid function and its inverse
def sigm1(x):
    return 1 / (1 + np.exp(-x))

def isigm1(y):
    # Clip y to avoid division by zero or log(0)
    y = np.clip(y, 0.001, 0.999)
    return -np.log(1 / y - 1)

# RBF layer training
def calc_phi(x, xi, n, nn, c, vars):
    phi = np.zeros((nn, n))  # Initialize

    for i in range(nn):
        for j in range(n):
            dr = 0
            for k in range(vars):
                dr += (xi[i, k] - x[j, k]) ** 2
            phi[i, j] = np.exp(-dr / c)  # Exponential
    
    return phi

def train_layer_1_rbf(neurons, vars, obs, n_per_part, inds_all, x, y, cc):
    a_all = []
    ooops = 0
    
    for i in range(neurons):
        print(f"Training neuron {i+1}/{neurons}")
        try:
            x1 = x[inds_all[n_per_part[i]:n_per_part[i+1]], :]
            obs1 = x1.shape[0]
            
            if obs1 == 0:
                raise ValueError("No samples assigned to neuron")
            
            phi1 = calc_phi(x1, x1, obs1, obs1, cc, vars)
            atmp = np.linalg.solve(phi1, y[inds_all[n_per_part[i]:n_per_part[i+1]]])
            
            if np.isnan(atmp).any() or np.isinf(atmp).any():
                a_all.append(np.zeros(obs1))
                ooops += 1
            else:
                a_all.append(atmp)
        except Exception as ex:
            ooops += 1
            print(f"Error in neuron {i}: {ex}")
            a_all.append(np.zeros(1))
    
    layer1 = np.zeros((obs, neurons))
    
    for i in range(neurons):
        print(f"Computing output for neuron {i+1}/{neurons}")
        x1 = x[inds_all[n_per_part[i]:n_per_part[i+1]], :]
        obs1 = x1.shape[0]
        
        for ii in range(obs):
            phis = sum(np.exp(-np.sum((x[ii, :] - x1[j, :]) ** 2) / cc) * a_all[i][j] for j in range(obs1))
            layer1[ii, i] = phis
    
    print("Training output layer...")
    X_layer1 = np.hstack((layer1, np.ones((obs, 1))))
    a_layer1 = np.linalg.lstsq(X_layer1, y, rcond=None)[0]
    print("Training complete.")
    
    return a_all, a_layer1, layer1

# Sigmoid-based layer 1 training
def train_layer_1_sigmoid(neurons, vars, obs, n_per_part, inds_all, x, y):
    a_all = []
    ooops = 0

    for i in range(neurons):
        try:
            if i + 1 >= len(n_per_part):  
                print(f"Neuron {i}: Partition index out of range. Appending zero weights.")
                a_all.append(np.zeros(vars + 1))
                continue  

            current_indices = inds_all[n_per_part[i]:n_per_part[i + 1]]
            x_slice = x[current_indices]
            y_slice = y[current_indices]

            atmp = np.linalg.lstsq(
                np.hstack((np.ones((x_slice.shape[0], 1)), x_slice)),
                isigm1(y_slice), rcond=None
            )[0]

            if np.sum(np.isnan(atmp)) == 0 and np.sum(np.isinf(atmp)) == 0:
                a_all.append(atmp)
            else:
                ooops += 1
                print(f"Neuron {i}: Invalid weights (NaN or Inf). Appending zero weights.")
                a_all.append(np.zeros(vars + 1))
        except Exception as ex:
            ooops += 1
            print(f"Problem Neuron = {i}, Total Ooops = {ooops}, Ex = {ex}")
            a_all.append(np.zeros(vars + 1))
    
    print(f"Completed processing all neurons. Total Ooops = {ooops}.")
    print(f"a_all length = {len(a_all)}, expected = {neurons}")

    layer1 = np.zeros((obs, neurons))

    for i in range(neurons):
        if i % 100 == 0:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"{current_time} Applying sigmoid on {i} neuron, of {neurons} Total")

        layer1[:, i] = sigm1(np.dot(np.hstack((np.ones((obs, 1)), x)), a_all[i]))

    a_layer1 = np.linalg.lstsq(np.hstack((layer1, np.ones((obs, 1)))), y, rcond=None)[0]

    return a_all, a_layer1, layer1

# Fit function using n-fold cross-validation
def fit_nfolds_sigmoid(xx_train, yy_train, nof_folds, vars, i_train, i_fold, neurons):
    maetrs = []
    a_all_all = []
    a_layer1_all = []
    n_per_part_all = []
    inds_all_all = []
    xx_fold_all = []
    neurons_all = []

    for i in range(nof_folds):

        inds_fold = np.random.permutation(i_train)[:i_fold]
        xx_fold = xx_train[inds_fold, :]
        yy_fold = yy_train[inds_fold]
        neurons_all.append(neurons)

        # Unpack all three values from clustering
        inds_all_vec_unvec, n_per_part_new, items_per_neuron_new = ___clustering(neurons, xx_fold, 300)
        
        # Train the model with the updated clustering results
        a_all, a_layer1, layer1 = train_layer_1_sigmoid(neurons, vars, i_fold, n_per_part_new, inds_all_vec_unvec, xx_fold, yy_fold)

        a_all_all.append(a_all)
        a_layer1_all.append(a_layer1)
        n_per_part_all.append(n_per_part_new)
        inds_all_all.append(inds_all_vec_unvec)

        predl1 = np.dot(np.hstack([layer1, np.ones((i_fold, 1))]), a_layer1)

        maetr = np.mean(np.abs(yy_fold - predl1))
        maetrs.append(maetr)

        print(f"{datetime.now().strftime('%H:%M:%S')} {i} maetr={maetr}")
    
    return maetrs, a_all_all, a_layer1_all, n_per_part_all, inds_all_all, xx_fold_all, neurons_all

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten the images (28x28) to 1D arrays (28*28 = 784)
x_train_flat = x_train.reshape(-1, 28*28)

# Scale the data (important for K-Means, since it's sensitive to scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)

# Convert labels to binary for simplicity
y_train_bin = (y_train == 1).astype(int)  # Binary classification for digit 1

# Number of neurons and other parameters
neurons = 200  # Number of neurons (clusters)
vars = 784  # Number of features (flattened image pixels)
nof_folds = 5  # Number of folds for cross-validation
i_train = len(x_train)  # Number of training samples
i_fold = i_train // nof_folds  # Samples per fold

# Run the fitting function for n-fold cross-validation
maetrs, a_all_all, a_layer1_all, n_per_part_all, inds_all_all, xx_fold_all, neurons_all = fit_nfolds_sigmoid(x_train_scaled, y_train_bin, nof_folds, vars, i_train, i_fold, neurons)

# Display the mean absolute error (maetrs) from the cross-validation
print("Mean absolute errors (maetrs) per fold:", maetrs)
