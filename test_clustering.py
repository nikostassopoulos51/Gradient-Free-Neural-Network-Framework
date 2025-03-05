import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf  # To load MNIST
from sklearn.preprocessing import StandardScaler

def ___clustering(neurons, xx_clust, max_iter):
    kmeans = KMeans(n_clusters=neurons, max_iter=max_iter, n_init=10, random_state=51)
    kmeans.fit(xx_clust)  # Perform clustering on input matrix xx_clust

    assignments = kmeans.labels_  # Assign clusters for each sample
    counts = np.bincount(assignments)  # Count sample items per cluster

    sorted_indices = np.argsort(assignments)  # Sort the matrix assignments in ascending order & return indices that would sort the matrix
    inds_all = sorted_indices.tolist()  # Convert the NumPy Array sorted_indices into Python List
    items_per_neuron = counts.tolist()
    n_per_part = [0] + np.cumsum(items_per_neuron).tolist()  # Find the cumulative sum of the array items/neuron and turn it into a Python List

    return inds_all, n_per_part, items_per_neuron

# Load MNIST dataset (train and test sets)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Flatten the images (28x28) to 1D arrays (28*28 = 784)
x_train_flat = x_train.reshape(-1, 28*28)

# Optionally scale the data (important for K-Means, since it's sensitive to scaling)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)

# Now we can pass the MNIST data to the clustering function
neurons = 10  # Since MNIST has 10 unique digit classes, we might want 10 clusters
max_iter = 100  # Maximum number of iterations

# Call the clustering function
inds_all, n_per_part, items_per_neuron = ___clustering(neurons, x_train_scaled, max_iter)

# Output the results
print("Sorted Indices (Data points per cluster):", inds_all)
print("Cumulative number of items per neuron:", n_per_part)
print("Items per neuron (cluster size):", items_per_neuron)