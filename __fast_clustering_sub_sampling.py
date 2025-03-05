'''
To perform sub-sampling & clustering for each digit (0-9).
'''

import time
import numpy as np # type: ignore

neurons1 = 200  # To initialize number of clusters per digit
inds_all = []  # List to store indices
items_per_neuron = []  # List to store number of items/cluster

for i in range(10):

    start_time = time.time()
    print(f"{i}: {time.strftime(' %H : %M : %S')}")

    i1 = (yy_train_all == i) # We create a bollean-logic array, in which each element is True, if the corresponding element of yy_train_all is equal to current digit i (of the loop) # type: ignore

    inds_all1, n_per_part1, items_per_neuron1 = ___clustering(neurons1, xx_train[i1, :], 300) # Assign to these variables, the returning values of the clustering function <3 # type: ignore

    ia1 = np.arange(1, i_train + 1)[i1] # type: ignore # Create a Python Array of indices (from 1 to i_train), then select those where i1 is True
    inds_all.extend(ia1[inds_all1].tolist()) # Select indices from ia1 & append to inds.all array 
    items_per_neuron.extend(items_per_neuron1) # Append the cluster sizes - updates the list with the number of samples assigned to each neuron

n_per_part = np.concantenate(([0], np.cumsum(items_per_neuron))) # Find cumulative sum of items/neuron as a NumPy Array, then add 0 in its beginning (represent that no items are assigned before cluster)

neurons = 10 * neurons1 # Total No of neurons
plt.plot(items_per_neuron) # Create plot line of the item/neuron elements & show how many data points are assigned to each neuron in the clustering process # type: ignore
plt.show() # type: ignore 

print("\n", min(items_per_neuron)) # Min No of items/neuron