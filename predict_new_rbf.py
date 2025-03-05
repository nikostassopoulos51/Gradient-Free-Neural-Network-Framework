import numpy as np
from datetime import datetime

def predict_new_rbf(a_all, a_layer1, x_new, obs, neurons, n_per_part, inds_all, x_old, vars_, cc):

    print("hello")
    
    layer1 = np.zeros((obs, neurons))

    for i in range(neurons):
        if i in range(neurons):
            print(datetime.now().strftime("%H:%M:%S"), f" Predicting response of {i} neuron, of {neurons} Total.")

        x1 = np.copy(x_old[ inds_all[n_per_part[i] + 1 : n_per_part[i + 1]], : ])
        obs1 = x1.shape[0]

        '''
        n_per_part[i] + 1 : n_per_part[i + 1] --> a slice of indices from the array n_per_part
        inds[slice of indices] --> a subset of indices, based on the slice
        x_old[inds[slice], :] --> extract rows, based on the given subset of the indices
        create a copy, cause we don't want to affect the original x_old
        '''

        phi1 = calc_phi(x1, x_new, obs1, obs, cc, vars_) # type: ignore
        layer1[:, i] = phi1 * a_all[i]

    predl1 = np.hstack((layer1, np.ones((obs, 1)))) @ a_layer1

    return predl1, layer1