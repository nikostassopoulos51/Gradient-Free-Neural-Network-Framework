import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

def clustering(neurons, xx_clust, max_iter=300):
    kmeans = KMeans(n_clusters=neurons, max_iter=max_iter, n_init=10, random_state=51)
    kmeans.fit(xx_clust)
    assignments = kmeans.labels_
    counts = np.bincount(assignments)
    sorted_indices = np.argsort(assignments)
    inds_all = sorted_indices.tolist()
    items_per_neuron = counts.tolist()
    n_per_part = [0] + np.cumsum(items_per_neuron).tolist()
    return inds_all, n_per_part, items_per_neuron

def sigm1(x):
    x = np.clip(x, -100, 100)
    return 1 / (1 + np.exp(-x))

def isigm1(y):
    y = np.clip(y, 0.001, 0.999)
    return -np.log(1 / y - 1)

def train_layer_1_sigmoid(neurons, vars, obs, n_per_part, inds_all, x, y):
    a_all = []
    for i in range(neurons):
        try:
            current_indices = inds_all[n_per_part[i]: min(n_per_part[i + 1], len(x))]
            x_slice = x[current_indices]
            y_slice = y[current_indices]
            y_slice = y_slice.reshape(-1, 1)  # Ensure y_slice is 2D
            atmp = np.linalg.lstsq(np.hstack((np.ones((x_slice.shape[0], 1)), x_slice)), isigm1(y_slice), rcond=None)[0]
            a_all.append(atmp if not np.isnan(atmp).any() else np.zeros(vars + 1))
        except:
            a_all.append(np.zeros(vars + 1))
    
    a_all = np.array(a_all)
    if a_all.shape[0] != neurons:
        raise ValueError(f"Unexpected shape of a_all: {a_all.shape}")
    layer1 = sigm1(np.dot(np.hstack((np.ones((obs, 1)), x)), a_all.T))
    
    y = y.reshape(obs, -1)  # Ensure y is 2D
    a_layer1 = np.linalg.lstsq(np.hstack((layer1, np.ones((obs, 1)))), y, rcond=None)[0]
    if a_layer1.shape[0] != neurons + 1:
        raise ValueError(f"Expected a_layer1 to have {neurons + 1} coefficients, but got {a_layer1.shape[0]}")
    
    print("Sigmoid training completed.")
    return a_all, a_layer1, layer1

def fit_nfolds_sigmoid(xx_train, yy_train, nof_folds, vars, i_train, i_fold, neurons):
    maetrs, a_all_all, a_layer1_all = [], [], []
    for _ in range(nof_folds):
        inds_fold = np.random.permutation(i_train)[:i_fold]
        xx_fold, yy_fold = xx_train[inds_fold], yy_train[inds_fold]
        inds_all, n_per_part, _ = clustering(neurons, xx_fold)
        a_all, a_layer1, _ = train_layer_1_sigmoid(neurons, vars, i_fold, n_per_part, inds_all, xx_fold, yy_fold)
        a_all_all.append(a_all)
        a_layer1_all.append(a_layer1)
        predl1 = sigm1(np.dot(np.hstack((sigm1(np.dot(np.hstack((xx_fold, np.ones((i_fold, 1)))), a_all.T)), np.ones((i_fold, 1)))), a_layer1))
        maetrs.append(np.mean(np.abs(yy_fold - predl1)))
    return maetrs, a_all_all, a_layer1_all

def predict(x_input, a_all, a_layer1):
    x_input = x_input.reshape(1, -1)
    layer1_out = sigm1(np.dot(np.hstack((np.ones((1, 1)), x_input)), a_all.T))
    output = sigm1(np.dot(np.hstack((layer1_out, np.ones((1, 1)))), a_layer1))
    return np.argmax(output)

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(-1, 28 * 28)
x_train_scaled = StandardScaler().fit_transform(x_train_flat)
y_train_one_hot = np.eye(10)[y_train]

unique, counts = np.unique(y_train, return_counts=True)
print(f"Digit distribution in dataset: {dict(zip(unique, counts))}")

neurons, vars, nof_folds = 200, 784, 5
i_train, i_fold = len(x_train), len(x_train) // nof_folds

maetrs, a_all_all, a_layer1_all = fit_nfolds_sigmoid(x_train_scaled, y_train_one_hot, nof_folds, vars, i_train, i_fold, neurons)

print("Mean absolute errors per fold:", maetrs)
plt.plot(maetrs, marker='o')
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("Model Performance Across Folds")
plt.show()

test_sample = x_train_scaled[0]
true_label = y_train[0]
predicted_label = predict(test_sample, a_all_all[0], a_layer1_all[0])

print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
