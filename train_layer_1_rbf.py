import numpy as np

def train_layer_1_rbf(neurons, vars, obs, n_per_part, inds_all, x, y, cc):
    
    a_all = []  # List to store weights for each neuron
    ooops = 0   # Counter for failed weight calculations


    # Iterate over each neuron, select data slices & compute weights using the RBF kernel!

    for i in range(neurons):

        print(f"Calculating weights w for {i} neuron, of {neurons} Total.")

        try:
            # Select data slice for the current neuron
            x1 = x[inds_all[n_per_part[i]:n_per_part[i+1]], :]
            obs1 = x1.shape[0]
            
            # Compute the RBF kernel matrix (phi1)
            phi1 = calc_phi(x1, x1, obs1, obs1, cc, vars) # type: ignore
            
            # Solve for weights using the RBF kernel
            atmp = np.linalg.solve(phi1, y[inds_all[n_per_part[i]:n_per_part[i+1]]])
            
            # Check for valid weights
            if np.isnan(atmp).any() or np.isinf(atmp).any():
                a_all.append(np.zeros(obs1))
                ooops += 1
            else:
                a_all.append(atmp)

        except Exception as ex:
            ooops += 1
            print(f"Error in neuron {i}: {ex}")


    # Output for each neuron.

    layer1 = np.zeros((obs, neurons)) #Initialize

    for i in range(neurons):
        print(f"Calculating output of neuron {i} , of {neurons} Total.")
        
        # Slice the data for the current neuron
        x1 = x[inds_all[n_per_part[i]:n_per_part[i + 1]], :]
        obs1 = x1.shape[0]
        
        # Compute the output for each neuron
        for ii in range(obs):
            phis = 0.0
            for j in range(obs1):
                dr = 0
                for k in range(vars):
                    dr += (x[ii, k] - x1[j, k]) ** 2
                phis += np.exp(-dr / cc) * a_all[i][j]   # RBF kernel computed as exp(-dr / cc) where dr is squared Euclidean distance between input data points, just like in theoretical part!
            layer1[ii, i] = phis


    # Weights of the output layer - use least squares!

    print(f"Solving system ({obs} by {neurons + 1}) for weights v in the output Layer")
    
    X_layer1 = np.hstack((layer1, np.ones((obs, 1))))  # add bias!
    a_layer1 = np.linalg.lstsq(X_layer1, y, rcond = None)[0]
    
    print("All weights computed.")
    
    return a_all, a_layer1, layer1


