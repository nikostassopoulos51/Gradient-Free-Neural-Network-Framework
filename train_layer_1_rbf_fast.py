import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Optimized RBF Calculation (Vectorized)
def calc_phi(x, xi, c):
    diff = x[:, np.newaxis, :] - xi[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    return np.exp(-dist_sq / c)

# Clustering function
def clustering(neurons, xx_clust, max_iter=300):
    kmeans = KMeans(n_clusters=neurons, max_iter=max_iter, n_init=10, random_state=51)
    kmeans.fit(xx_clust)

    assignments = kmeans.labels_
    counts = np.bincount(assignments)

    sorted_indices = np.argsort(assignments).tolist()
    n_per_part = [0] + np.cumsum(counts).tolist()

    return sorted_indices, n_per_part, counts.tolist()

# Optimized RBF Fast Training
def train_layer_1_rbf_fast(neurons, vars, obs, n_per_part, inds_all, x, y, cc):
    a_all = []
    ooops = 0

    print("Training RBF FAST layer...")

    for i in range(neurons):
        try:
            x1 = x[inds_all[n_per_part[i]:n_per_part[i+1]], :]
            y1 = y[inds_all[n_per_part[i]:n_per_part[i+1]]]
            
            if x1.shape[0] == 0:
                raise ValueError("No samples assigned to neuron")

            phi1 = calc_phi(x1, x1, cc)
            atmp = np.linalg.lstsq(phi1, y1, rcond=None)[0]

            if np.isnan(atmp).any() or np.isinf(atmp).any():
                a_all.append(np.zeros_like(atmp))
                ooops += 1
            else:
                a_all.append(atmp)

        except Exception as ex:
            ooops += 1
            print(f"Error in neuron {i}: {ex}")
            a_all.append(np.zeros(1))

    # Compute RBF activations for all neurons
    print("Computing activations...")
    layer1 = np.zeros((obs, neurons))

    for i in range(neurons):
        x1 = x[inds_all[n_per_part[i]:n_per_part[i+1]], :]
        phi = calc_phi(x, x1, cc)
        layer1[:, i] = phi @ a_all[i]

    # Train output layer
    print(f"Solving system ({obs} x {neurons + 1}) for output weights...")
    X_layer1 = np.hstack((layer1, np.ones((obs, 1))))
    a_layer1 = np.linalg.lstsq(X_layer1, y, rcond=None)[0]

    print("Training complete.")
    return a_all, a_layer1, layer1

# Load & Preprocess MNIST
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(-1, 28 * 28)
x_train_scaled = StandardScaler().fit_transform(x_train_flat)

# Perform Clustering for Each Digit
neurons_per_digit = 20
total_neurons = 10 * neurons_per_digit

inds_all = []
items_per_neuron = []

for i in range(10):
    print(f"Processing digit {i}...")
    digit_indices = np.where(y_train == i)[0]
    inds_all1, n_per_part1, items_per_neuron1 = clustering(neurons_per_digit, x_train_scaled[digit_indices])

    inds_all.extend(digit_indices[inds_all1].tolist())
    items_per_neuron.extend(items_per_neuron1)

n_per_part = np.concatenate(([0], np.cumsum(items_per_neuron)))

# Train RBF FAST Network
cc = 1.0
a_all, a_layer1, layer1 = train_layer_1_rbf_fast(total_neurons, x_train_scaled.shape[1], len(x_train), n_per_part, inds_all, x_train_scaled, y_train, cc)

print("Layer 1 activations:")
print(layer1)
print("Output layer weights:")
print(a_layer1)
