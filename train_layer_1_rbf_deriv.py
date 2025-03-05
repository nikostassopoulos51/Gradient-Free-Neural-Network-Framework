import numpy as np

def train_layer_1_rbf_deriv(neurons, vars, obs, n_per_part, inds_all, x, y, cc):

    a_all = []  # List to store weights for each neuron
    ooops = 0   # Counter for failed weight calculations
    errs = []   # List to store errors for each computation
    obs1 = 50   # Number of observations to sample for each training iteration
    itmps = []  # List to store indices of selected observations


    # For each neuron, we select a subset of data - calculate RBF kernel & derivative - compute corresponding weights!

    for i in range(neurons):

        print(f"Calculating weights w for {i} neuron, of {neurons} Total.")
        
        # Randomly take `obs1` indices from observations
        itmp = np.random.permutation(obs)[:obs1]
        x1 = x[itmp, :]
        
        # Compute RBF kernel & its derivative
        phi1 = calc_phi(x1, x1, obs1, obs1, cc, vars) # type: ignore
        phi1_deriv = calc_phi_deriv(x1, x1, obs1, obs1, cc, vars, np.random.randint(1, vars)) # type: ignore
        
        # Solve for weights
        atmp = np.linalg.lstsq(np.vstack([phi1, phi1_deriv]), np.hstack([y[itmp], np.zeros(obs1)]), rcond = None)[0]
        
        # Calculate error
        err1 = np.mean(np.abs(np.dot(phi1, atmp) - y[itmp]))
        
        # If weights valid, store them
        if not np.isnan(atmp).any() and not np.isinf(atmp).any() and np.sum(np.abs(atmp)) > 0:
            a_all.append(atmp)
            itmps.append(itmp)
            errs.append(err1)
        else:
            ooops += 1

        print(f"ooops={ooops}")


    # Now sort neurons by error - select top 2000

    ii1 = np.argsort(errs)[:2000]
    a_all = [a_all[i] for i in ii1]
    itmps = [itmps[i] for i in ii1]

    return a_all, itmps