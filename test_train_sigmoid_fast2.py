import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def sigm1(x):
    return 1 / (1 + np.exp(-x))

def isigm1(x):
    return np.log(x / (1 - x))

def train_layer_1_sigmoid_fast2(neurons, vars, obs, n_per_part, inds_all, x, y):
    a_all = []  
    ooops = 0    
    i_keep = []  
    errs = []    
    obs1 = 13  

    for i in range(neurons):
        try:
            if i + 1 >= len(n_per_part):
                a_all.append(np.zeros((vars + 1, 1)))
                continue  

            current_indices = inds_all[n_per_part[i]:n_per_part[i + 1]]
            if len(current_indices) < obs1:
                continue  

            x_slice = x[current_indices]
            y_slice = y[current_indices]

            atmp = np.linalg.lstsq(np.hstack((np.ones((x_slice.shape[0], 1)), x_slice)), isigm1(y_slice), rcond=None)[0]

            err1 = np.mean(np.abs(sigm1(np.hstack((np.ones((x_slice.shape[0], 1)), x_slice)) @ atmp) - y_slice))
            errs.append(err1)

            if np.sum(np.isnan(atmp)) == 0 and np.sum(np.isinf(atmp)) == 0 and np.sum(np.abs(atmp[1:])) > 0: 
                a_all.append(atmp.reshape(-1, 1))
                i_keep.append(i)
            else:
                ooops += 1

        except Exception as ex:
            ooops += 1
            print(f"Problem neuron={i}, total ooops={ooops}, ex={ex}")
            a_all.append(np.zeros((vars + 1, 1)))  

    if len(errs) > 0:
        ii1 = np.argsort(errs)[:min(len(errs), 20000)]
        a_all = [a_all[i] for i in ii1]  

    X = np.hstack((np.ones((obs, 1)), x))  
    A_all = np.hstack(a_all) if a_all else np.zeros((vars + 1, neurons))

    print(f"Shape of X: {X.shape}")
    print(f"Shape of A_all: {A_all.shape}")

    layer1 = sigm1(X @ A_all)

    cors = np.zeros(layer1.shape[1])
    for i in range(len(cors)):
        cors[i] = np.mean(np.abs(layer1[:, i] - y))

    ico = np.argsort(np.abs(cors))[:min(len(cors), 20000)]
    layer1 = layer1[:, ico]
    a_all = [a_all[i] for i in ico]

    layer1_aug = np.hstack((layer1, np.ones((obs, 1))))
    a_layer1 = np.linalg.pinv(layer1_aug.T @ layer1_aug) @ layer1_aug.T @ y

    return a_all, a_layer1, layer1


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

a_all, a_layer1, layer1 = train_layer_1_sigmoid_fast2(neurons, x_train_scaled.shape[1], obs, n_per_part, inds_all, x_train_scaled[:obs], y)

print("Weights for each neuron (a_all):")
for i, weights in enumerate(a_all):
    print(f"Neuron {i}: {weights[:5]} ...")

print("\nOutput layer weights (a_layer1):")
print(a_layer1)

print("\nFirst layer output (layer1):")
print(layer1)
