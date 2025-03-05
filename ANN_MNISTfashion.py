import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def clustering(neurons, xx_clust, max_iter=300):
    kmeans = KMeans(n_clusters=neurons, max_iter=max_iter, n_init=20, random_state=51)
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
    y = np.clip(y, 0.01, 0.99)
    
    for i in range(neurons):
        try:
            current_indices = inds_all[n_per_part[i] : n_per_part[i + 1]]
            if len(current_indices) == 0:
                a_all.append(np.zeros(vars + 1))
                continue
            x_slice = x[current_indices]
            y_slice = y[current_indices]

            atmp = np.linalg.lstsq(
                np.hstack((np.ones((x_slice.shape[0], 1)), x_slice)), isigm1(y_slice), rcond=None
            )[0]
            
            atmp = atmp.flatten()
            if atmp.shape[0] == vars + 1:
                a_all.append(atmp)
            else:
                a_all.append(np.zeros(vars + 1))
        except:
            a_all.append(np.zeros(vars + 1))
    
    a_all = np.array(a_all)
    expected_size = (neurons, vars + 1)
    actual_size = a_all.shape
    
    if a_all.size == expected_size[0] * expected_size[1]:  
        a_all = a_all.reshape(expected_size)
    else:
        raise ValueError(f"Incorrect shape: {actual_size}. Check a_all.")
    
    layer1 = sigm1(np.dot(np.hstack((np.ones((obs, 1)), x)), a_all.T))
    a_layer1 = np.linalg.lstsq(
        np.hstack((layer1, np.ones((obs, 1)))), y, rcond=None
    )[0]
    
    return a_all, a_layer1, layer1

def fit_nfolds_sigmoid(xx_train, yy_train, nof_folds, vars, i_train, i_fold, neurons):
    np.random.seed(None)
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

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def predict(x_input, a_all, a_layer1):
    x_input = x_input.reshape(1, -1)
    layer1_out = sigm1(np.dot(np.hstack((np.ones((1, 1)), x_input)), a_all.T))
    output = np.dot(np.hstack((layer1_out, np.ones((1, 1)))), a_layer1)
    probabilities = softmax(output)
    return np.argmax(probabilities)

(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()
x_train_flat = x_train.reshape(-1, 28 * 28)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)

y_train_one_hot = np.eye(10)[y_train]

unique, counts = np.unique(y_train, return_counts=True)
print(f"Fashion item distribution in dataset: {dict(zip(unique, counts))}")

neurons, vars, nof_folds = 300, 784, 5
i_train, i_fold = len(x_train), len(x_train) // nof_folds

maetrs, a_all_all, a_layer1_all = fit_nfolds_sigmoid(x_train_scaled, y_train_one_hot, nof_folds, vars, i_train, i_fold, neurons)

print("Mean absolute errors per fold:", maetrs)
plt.plot(maetrs, marker='o')
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("Model Performance Across Folds")
plt.show()

random_idx = np.random.randint(0, len(x_train_scaled))
test_sample = x_train_scaled[random_idx]
true_label = y_train[random_idx]

predicted_label = predict(test_sample, a_all_all[0], a_layer1_all[0])

print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
