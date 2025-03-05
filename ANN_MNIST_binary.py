import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

# ----- CLUSTERING & SUBSAMPLING -----
def clustering(neurons, xx_clust, max_iter = 300):
    kmeans = KMeans(n_clusters = neurons, max_iter = max_iter, n_init = 10, random_state = 51)
    kmeans.fit(xx_clust)
    assignments = kmeans.labels_
    counts = np.bincount(assignments)
    sorted_indices = np.argsort(assignments)
    inds_all = sorted_indices.tolist()
    items_per_neuron = counts.tolist()

    # Print clustering statistics
    print(f"Clustering completed. Neurons: {neurons}")
    print(f"Items per neuron: {items_per_neuron}")
    print(f"Total samples clustered: {sum(items_per_neuron)}")

    n_per_part = [0] + np.cumsum(items_per_neuron).tolist()

    return inds_all, n_per_part, items_per_neuron

# ----- SIGMOID FUNCTIONS -----
def sigm1(x):
    x = np.clip(x, -100, 100) 
    return 1 / (1 + np.exp(-x))

def isigm1(y):
    y = np.clip(y, 0.001, 0.999)
    return -np.log(1 / y - 1)

# ----- TRAINING WITH SIGMOID -----
def train_layer_1_sigmoid(neurons, vars, obs, n_per_part, inds_all, x, y):
    a_all = []
    for i in range(neurons):
        try:
            current_indices = inds_all[n_per_part[i] : n_per_part[i + 1]]
            x_slice = x[current_indices]
            y_slice = y[current_indices]
            atmp = np.linalg.lstsq(np.hstack((np.ones((x_slice.shape[0], 1)), x_slice)), isigm1(y_slice), rcond = None)[0]
            a_all.append(atmp if not np.isnan(atmp).any() else np.zeros(vars + 1))
        except:
            a_all.append(np.zeros(vars + 1))
    layer1 = sigm1(np.dot(np.hstack((np.ones((obs, 1)), x)), np.vstack(a_all).T))
    
    a_layer1 = np.linalg.lstsq(np.hstack((layer1, np.ones((obs, 1)))), y, rcond = None)[0]
    if a_layer1.shape[0] != neurons + 1:
        raise ValueError(f"Expected a_layer1 to have {neurons + 1} coefficients, but got {a_layer1.shape[0]}")
    
    # Print completion of sigmoid training
    print("Sigmoid training completed.")

    # Print output weights after sigmoid training
    print(f"Final output weights (Sigmoid): {a_layer1}")

    return a_all, a_layer1, layer1

# ----- TRAINING WITH RBF -----
def train_layer_1_rbf(neurons, vars, obs, n_per_part, inds_all, x, y, cc=1.0):
    a_all = []
    for i in range(neurons):
        try:
            x1 = x[inds_all[n_per_part[i] : n_per_part[i+1]], :]
            obs1 = x1.shape[0]
            phi1 = np.exp(-np.linalg.norm(x1[:, None] - x1, axis=2) ** 2 / cc)
            atmp = np.linalg.solve(phi1, y[inds_all[n_per_part[i]:n_per_part[i+1]]])
            a_all.append(atmp if not np.isnan(atmp).any() else np.zeros(obs1))
        except:
            a_all.append(np.zeros(1))
    
    # Print final RBF weights
    if 'a_all' in locals():  # Ensure the variable exists before printing
        print(f"Final RBF weights: {a_all}")
    
    return a_all

# ----- NFITFOLDS WITH SIGMOID -----
def fit_nfolds_sigmoid(xx_train, yy_train, nof_folds, vars, i_train, i_fold, neurons):
    maetrs, a_all_all, a_layer1_all = [], [], []
    for i in range(nof_folds):
        inds_fold = np.random.permutation(i_train)[:i_fold]
        xx_fold, yy_fold = xx_train[inds_fold], yy_train[inds_fold]
        inds_all, n_per_part, _ = clustering(neurons, xx_fold)
        a_all, a_layer1, _ = train_layer_1_sigmoid(neurons, vars, i_fold, n_per_part, inds_all, xx_fold, yy_fold)
        a_all_all.append(a_all)
        a_layer1_all.append(a_layer1)
        predl1 = sigm1(np.dot(np.hstack((sigm1(np.dot(np.hstack((xx_fold, np.ones((i_fold, 1)))), np.vstack(a_all).T)), np.ones((i_fold, 1)))), a_layer1))
        maetrs.append(np.mean(np.abs(yy_fold - predl1)))
    return maetrs, a_all_all, a_layer1_all

def predict(x_input, a_all, a_layer1):
    """
    Predict the digit class for a given input using trained sigmoid model.
    
    Parameters:
        x_input (numpy array): Flattened 28x28 image, shape (784,)
        a_all (list of numpy arrays): Trained weights for the first layer
        a_layer1 (numpy array): Trained weights for the output layer
    
    Returns the predicted digit (0 or 1 in this case)
    """

    x_input = x_input.reshape(1, -1)  # Ensure input is 2D
    layer1_out = sigm1(np.dot(np.hstack((np.ones((1, 1)), x_input)), np.vstack(a_all).T))
    output = sigm1(np.dot(np.hstack((layer1_out, np.ones((1, 1)))), a_layer1))
    return int(output >= 0.5)  # Binary classification (1 if >= 0.5, else 0)

(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train_flat = x_train.reshape(-1, 28*28)
x_train_scaled = StandardScaler().fit_transform(x_train_flat)
y_train_bin = (y_train == 1).astype(int)  # Binary classification for digit '1'

# Print subsampling digits
unique, counts = np.unique(y_train_bin, return_counts=True)
print(f"Subsampling distribution: {dict(zip(unique, counts))}")
    
# Parameters
neurons, vars, nof_folds = 200, 784, 5
i_train, i_fold = len(x_train), len(x_train) // nof_folds
    
# Train with N-fold cross-validation
maetrs, a_all_all, a_layer1_all = fit_nfolds_sigmoid(x_train_scaled, y_train_bin, nof_folds, vars, i_train, i_fold, neurons)
    
# Results
print("Mean absolute errors per fold:", maetrs)
plt.plot(maetrs, marker='o')
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("Model Performance Across Folds")
plt.show()

# Select a test sample from the training set
test_sample = x_train_scaled[0]
true_label = y_train_bin[0]

# Use trained weights from the first fold (you can modify this for ensembles)
predicted_label = predict(test_sample, a_all_all[0], a_layer1_all[0])

# Print prediction result
print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
