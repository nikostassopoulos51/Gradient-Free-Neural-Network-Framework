import numpy as np

def train_layer_1_sigmoid_fast(neurons, vars, obs, n_per_part, inds_all, x, y):

    a_all = []
    ooops = 0

    for i in range(neurons):
        try:
            atmp = np.linalg.solve(
                np.hstack((
                    np.ones((n_per_part[i + 1] - n_per_part[i],)), 
                    x[inds_all[n_per_part[i] + 1:n_per_part[i + 1]], :]
                )),
                isigm1(y[inds_all[n_per_part[i] + 1:n_per_part[i + 1]]])  # type: ignore
            )

            if np.sum(np.isnan(atmp)) == 0 and np.sum(np.isinf(atmp)) == 0 and np.sum(np.abs(atmp[1:])) > 0:
                a_all.append(atmp)  # Append valid weights

            else:
                ooops += 1
            
        except Exception as ex:
            ooops += 1
            print(f"Problem Neuron = {i}, Total Ooops = {ooops}, Ex = {ex}")
            a_all.append(np.zeros(vars + 1))
    
    layer1 = sigm1(np.dot(np.hstack((np.ones((obs, 1)), x)), np.column_stack(a_all)))  # type: ignore
    mat1 = np.linalg.inv(np.dot(layer1.T, np.hstack((layer1, np.ones((obs, 1))))))
    a_layer1 = np.dot(mat1, np.dot(layer1.T, y))

    return a_all, a_layer1, layer1, mat1

