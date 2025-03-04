import numpy as np

def train_layer_1_sigmoid_fast2(neurons, vars, obs, n_per_part, inds_all, x, y):

    a_all = []  # List to store weights for each neuron.
    ooops = 0    # Counter for failed weight calculations.
    i_keep = []  # List to keep indices of valid neurons.
    errs = []    # List to store errors for each neuron.
    obs1 = 13    # Random subset size for each neuron.
    i = 0

    # Iterate over each neuron to calculate its weight vector based on the random subset of observations.

    for i in range(neurons):
        try:
            # Select a random subset of observations
            itmp = np.random.permutation(obs)[:obs1]
            
            # Solve for the weights (inverse of the sigmoid)
            atmp = np.linalg.lstsq(np.hstack((np.ones((obs1, 1)), x[itmp, :])), isigm1(y[itmp]), rcond=None)[0] # type: ignore
            
            # Calculate the error (mean absolute error)
            err1 = np.mean(np.abs(sigm1(np.hstack((np.ones((obs1, 1)), x[itmp, :]))) @ atmp - y[itmp])) # type: ignore
    

            # Valid Neurons & Error Handling. If the computed weights are valid (not NaN or Inf), they are stored. 
            # Otherwise, the number of failed attempts (ooops) is incremented

            if np.sum(np.isnan(atmp)) == 0 and np.sum(np.isinf(atmp)) == 0 and np.sum(np.abs(atmp[1:])) > 0: 
                a_all.append(atmp)
                errs.append(err1)
                i_keep.append(i)

            else:
                ooops += 1

        except Exception as ex:    # For cases where the weight calculation fails - appends 0 to the list for weights.
            ooops += 1
            print(f"problem neuron={i} total ooops={ooops} ex={ex}")
            a_all.append(np.zeros(vars + 1))


    # The best neurons, with smallest errors, are selected ato construct the 1st layer :)

    ii1 = np.argsort(errs)[:20000]  # Sort neurons by error.
    a_all = [a_all[i] for i in ii1] # Select neurons. 
    layer1 = sigm1(np.hstack((np.ones((obs, 1)), x)) @ np.hstack(a_all)) # Build 1st layer - applies sigmoid to the weightened sum of inputs # type: ignore


    # Select top 20.000 features & update list of weights & 1st layer
    
    cors = np.zeros(layer1.shape[1])
    
    for i in range(len(cors)):
        cors[i] = np.mean(np.abs(layer1[:, i] - y))
    
    ico = np.argsort(np.abs(cors))[:20000]
    layer1 = layer1[:, ico]
    a_all = [a_all[i] for i in ico]


    # Solve for output layer weights - use least squares solution!

    a_layer1 = np.linalg.inv(np.hstack((layer1, np.ones((obs, 1)))).T @ np.hstack((layer1, np.ones((obs, 1))))) @ np.hstack((layer1, np.ones((obs, 1)))).T @ y

    return a_all, a_layer1, layer1 









