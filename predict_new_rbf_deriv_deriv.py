import numpy as np

def predict_new_rbf_deriv_deriv(a_all, a_layer1, x_new, obs, neurons, n_per_part, inds_all, x_old, vars_, cc, var_seq):
    
    layer1 = np.zeros((obs, neurons))
    
    for i in range(neurons):  
        x1 = np.copy(x_old[inds_all[n_per_part[i] + 1 : n_per_part[i + 1]], :])
        obs1 = x1.shape[0]
        
        phi1 = calc_phi_deriv_deriv(x1, x_new, obs1, obs, cc, vars_, var_seq) # type: ignore
        layer1[:, i] = phi1 * a_all[i]
    
    predl1 = np.hstack((layer1, np.ones((obs, 1)))) @ a_layer1
    
    return predl1
