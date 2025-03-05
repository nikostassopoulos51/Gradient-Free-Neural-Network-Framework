import numpy as np

def train_layer_1_rbf_new(neurons, vars, obs, n_per_part, inds_all, x, y, cc):

    a_all = []
    ooops = 0


    #Training loop for each neuron

    for i in range(neurons):
        
        print(f"Calculating weights w for {i+1} neuron, of {neurons} Total.")
        
        if n_per_part[i + 1] - n_per_part[i] != 0:

            try:
                # Random data - neuron
                x1 = x[inds_all[n_per_part[i] + 1 : n_per_part[i + 1]], :]
                obs1 = x1.shape[0]
            
                # RBF kernel
                phi1 = calc_phi(x1, x1, obs1, obs1, cc, vars) # type: ignore
            
                # Weights - least squares
                atmp = np.linalg.lstsq(phi1.T, y[inds_all[n_per_part[i] + 1:n_per_part[i + 1]]], rcond = None)[0]
            
                # Weights valid
                if not np.isnan(atmp).any() and not np.isinf(atmp).any():
                    a_all.append(atmp)
                else:
                    a_all.append(np.zeros(obs1))
                    ooops += 1

            except Exception as ex:
                ooops += 1
                print(f"Error at neuron {i}: {ex}")
    

    # Now output layer - applying waights of RBF kernel

    layer1 = np.zeros((obs, neurons))

    for i in range(neurons):

        print(f"Calculating output of neuron {i+1}, of {neurons} Total.")
    
        # Random data - neuron
        x1 = x[inds_all[n_per_part[i] + 1 : n_per_part[i + 1]], :]
        obs1 = x1.shape[0]
    
        # RBF kernel
        phi1 = calc_phi(x1, x, obs1, obs, cc, vars) # type: ignore
    
        # Output for this neuron
        layer1[:, i] = np.dot(phi1, a_all[i])

    
    print(f"Solving system ({obs} by {neurons+1}) for weights v in the output Layer")
    xxxx = np.hstack([layer1, np.ones((obs, 1))])
    a_layer1 = np.linalg.lstsq(xxxx, y, rcond = None)[0]

    return a_all, a_layer1, layer1