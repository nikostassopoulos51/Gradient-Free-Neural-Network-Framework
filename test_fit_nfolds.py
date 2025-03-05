import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import time
from datetime import datetime

def clustering(neurons, xx_clust, max_iter=300):
    kmeans = KMeans(n_clusters=neurons, max_iter=max_iter, n_init=10, random_state=51)
    kmeans.fit(xx_clust)
    
    assignments = kmeans.labels_
    counts = np.bincount(assignments)
    
    sorted_indices = np.argsort(assignments).tolist()
    n_per_part = [0] + np.cumsum(counts).tolist()
    
    return sorted_indices, n_per_part, counts.tolist()

def calc_phi(x, xi, n, nn, c, vars):
    phi = np.zeros((nn, n))  # Initialize

    for i in range(nn):
        for j in range(n):
            dr = 0
            for k in range(vars):
                dr += (xi[i, k] - x[j, k]) ** 2
            phi[i, j] = np.exp(-dr / c)  # Exponential
    
    return phi

def train_layer_1_rbf(neurons, x, y, n_per_part, inds_all, cc):
    a_all = []
    layer1 = np.zeros((len(y), neurons))
    
    for i in range(neurons):
        try:
            x1 = x[inds_all[n_per_part[i]:n_per_part[i+1]], :]
            y1 = y[inds_all[n_per_part[i]:n_per_part[i+1]]]
            
            if x1.shape[0] == 0:
                raise ValueError("No samples assigned to neuron")
            
            phi1 = calc_phi(x1, x1, cc)
            atmp = np.linalg.lstsq(phi1, y1, rcond=None)[0]
            a_all.append(atmp)
            
            phi = calc_phi(x, x1, cc)
            layer1[:, i] = phi @ atmp
        except Exception as ex:
            print(f"Error in neuron {i}: {ex}")
            a_all.append(np.zeros(1))
    
    X_layer1 = np.hstack((layer1, np.ones((len(y), 1))))
    a_layer1 = np.linalg.lstsq(X_layer1, y, rcond=None)[0]
    return a_all, a_layer1, layer1

def fit_nfolds(xx_train, yy_train, nof_folds, neurons, cc1, n_per_part, inds_all):
    kf = KFold(n_splits=nof_folds, shuffle=True, random_state=42)
    maetrs = []
    pred_folds_all = np.zeros((len(xx_train), nof_folds))
    
    for i, (train_idx, val_idx) in enumerate(kf.split(xx_train)):
        print(f"Training Fold {i+1}/{nof_folds}")
        
        xx_train_fold, xx_val_fold = xx_train[train_idx], xx_train[val_idx]
        yy_train_fold, yy_val_fold = yy_train[train_idx], yy_train[val_idx]
        
        a_all, a_layer1, layer1 = train_layer_1_rbf(neurons, xx_train_fold, yy_train_fold, n_per_part, inds_all, cc1)
        
        predl1 = np.hstack([layer1, np.ones((len(yy_val_fold), 1))]) @ a_layer1
        pred_folds_all[val_idx, i] = predl1
        
        maetr = np.mean(np.abs(yy_val_fold - predl1))
        maetrs.append(maetr)
        print(f"{datetime.now().strftime('%H:%M:%S')} Fold {i+1}, MAE: {maetr:.4f}")
    
    return maetrs, pred_folds_all

# Load MNIST dataset
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(-1, 28*28)
x_train_scaled = StandardScaler().fit_transform(x_train_flat)

# Perform clustering for each digit
neurons_per_digit = 20
inds_all = []
items_per_neuron = []

for i in range(10):
    print(f"Processing digit {i}...")
    digit_indices = np.where(y_train == i)[0]
    inds_all1, n_per_part1, items_per_neuron1 = clustering(neurons_per_digit, x_train_scaled[digit_indices])
    
    inds_all.extend(digit_indices[inds_all1].tolist())
    items_per_neuron.extend(items_per_neuron1)

n_per_part = np.concatenate(([0], np.cumsum(items_per_neuron)))

# Train using K-fold cross-validation
nof_folds = 5
cc1 = 1.0
maetrs, pred_folds_all = fit_nfolds(x_train_scaled, y_train, nof_folds, neurons_per_digit * 10, cc1, n_per_part, inds_all)

print("\nMAE for each fold:", maetrs)
