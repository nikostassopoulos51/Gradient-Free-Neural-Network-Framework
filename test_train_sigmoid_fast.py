import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf

def sigm1(x):
    return 1 / (1 + np.exp(-x))

def isigm1(y):
    return -np.log(1 / y - 1)

def train_layer_1_sigmoid_fast(neurons, vars, obs, n_per_part, inds_all, x, y):
    a_all = []
    ooops = 0

    for i in range(neurons):
        try:
            x_part = x[inds_all[n_per_part[i]:n_per_part[i + 1]], :]
            y_part = isigm1(y[inds_all[n_per_part[i]:n_per_part[i + 1]]])
            X = np.hstack((np.ones((x_part.shape[0], 1)), x_part))

            atmp, _, _, _ = np.linalg.lstsq(X, y_part, rcond=None)

            if np.sum(np.isnan(atmp)) == 0 and np.sum(np.isinf(atmp)) == 0 and np.sum(np.abs(atmp[1:])) > 0:
                a_all.append(atmp)
            else:
                ooops += 1
                a_all.append(np.zeros(vars + 1))

        except Exception as ex:
            ooops += 1
            print(f"Problem Neuron = {i}, Total Ooops = {ooops}, Ex = {ex}")
            a_all.append(np.zeros(vars + 1))

    layer1 = sigm1(np.dot(np.hstack((np.ones((obs, 1)), x)), np.column_stack(a_all)))  

    mat1 = np.dot(layer1.T, layer1)  
    reg_term = 1e-6 * np.eye(mat1.shape[0])  
    mat1 += reg_term

    a_layer1 = np.dot(np.linalg.pinv(mat1), np.dot(layer1.T, y))  

    return a_all, a_layer1, layer1, mat1

# Load & Preprocess MNIST
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(-1, 28 * 28)
x_train_scaled = StandardScaler().fit_transform(x_train_flat)

# Perform Clustering
def clustering(neurons, xx_clust, max_iter=300):
    kmeans = KMeans(n_clusters=neurons, max_iter=max_iter, n_init=10, random_state=51)
    kmeans.fit(xx_clust)

    assignments = kmeans.labels_
    counts = np.bincount(assignments)

    sorted_indices = np.argsort(assignments).tolist()
    items_per_neuron = counts.tolist()
    n_per_part = [0] + np.cumsum(items_per_neuron).tolist()

    return sorted_indices, n_per_part, items_per_neuron

# Use Clustering to Define Partitions
neurons = 10  
obs = 1000  

inds_all, n_per_part, _ = clustering(neurons, x_train_scaled[:obs])

np.random.seed(42)
y = sigm1(np.random.rand(obs))

a_all, a_layer1, layer1, mat1 = train_layer_1_sigmoid_fast(neurons, x_train_scaled.shape[1], obs, n_per_part, inds_all, x_train_scaled[:obs], y)

print("a_all (Weights for each neuron) [Truncated]:")
for i, weights in enumerate(a_all):
    print(f"Neuron {i}: {weights[:5]} ...")

print("\na_layer1 (Output layer weights):")
print(a_layer1)

print("\nlayer1 (Activated output of the first layer):")
print(layer1)

print("\nmat1 (Matrix used in calculating output layer weights):")
print(mat1)
