import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def isigm1(y):
    return -np.log(1 / y - 1)

# RBF Kernel Function
def calc_phi(x, xi, c):
    diff = x[:, np.newaxis, :] - xi[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    return np.exp(-dist_sq / c)

# Train first layer using sigmoid activation
def train_layer_1_sigmoid(neurons, vars, obs, n_per_part, inds_all, x, y):
    a_all = []
    ooops = 0

    for i in range(neurons):
        try:
            if i + 1 >= len(n_per_part):
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
                a_all.append(np.zeros(vars + 1))
                
        except Exception as ex:
            ooops += 1
            a_all.append(np.zeros(vars + 1))

    layer1 = np.zeros((obs, neurons))
    for i in range(neurons):
        layer1[:, i] = sigmoid(np.dot(np.hstack((np.ones((obs, 1)), x)), a_all[i]))

    a_layer1 = np.linalg.lstsq(np.hstack((layer1, np.ones((obs, 1)))), y, rcond=None)[0]
    
    return a_all, a_layer1, layer1

# Prediction for Sigmoid-based Model
def predict_sigmoid(a_all, a_layer1, x_new):
    obs = x_new.shape[0]
    a_all_stacked = np.column_stack(a_all)  # Ensure correct shape
    layer1 = sigmoid(np.hstack([np.ones((obs, 1)), x_new]) @ a_all_stacked)
    predl1 = np.hstack([layer1, np.ones((obs, 1))]) @ a_layer1  
    return predl1, layer1

# Prediction for RBF-based Model
def predict_rbf(a_all, a_layer1, x_new, x_train, cc, n_per_part, inds_all):
    obs = x_new.shape[0]
    neurons = len(a_all)
    layer1 = np.zeros((obs, neurons))
    for i in range(neurons):
        x1 = x_train[inds_all[n_per_part[i]:n_per_part[i+1]], :]
        phi = calc_phi(x_new, x1, cc)
        layer1[:, i] = phi @ a_all[i]
    predl1 = np.hstack([layer1, np.ones((obs, 1))]) @ a_layer1
    return predl1, layer1

# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

# Define Clustering Parameters
neurons = 10  
obs = 1000  
inds_all, n_per_part, _ = clustering(neurons, x_train[:obs])
np.random.seed(42)
y = sigmoid(np.random.rand(obs))

# Train Layer 1 using Sigmoid Activation
a_all, a_layer1, layer1 = train_layer_1_sigmoid(neurons, x_train.shape[1], obs, n_per_part, inds_all, x_train[:obs], y)

# Test sigmoid-based model
predictions_sigmoid, _ = predict_sigmoid(a_all, a_layer1, x_test[:10])
print("Sigmoid Model Predictions:", predictions_sigmoid)

# Dummy parameters for RBF-based model
cc = 0.1
n_per_part = [0, 5000, 10000]
inds_all = np.arange(x_train.shape[0])
np.random.shuffle(inds_all)
a_all_rbf = [np.random.randn(5000) for _ in range(100)]

# Test RBF-based model
predictions_rbf, _ = predict_rbf(a_all_rbf, a_layer1, x_test[:10], x_train, cc, n_per_part, inds_all)
print("RBF Model Predictions:", predictions_rbf)
