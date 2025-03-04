import numpy as np

def deep_nnm(tmp_weights, tmp_xx, nodes, obs, vars_, vars_depe):

    n_bias = np.sum(nodes) + vars_depe 
    w_bias = tmp_weights[-n_bias:] # Take biases away from weights, extract the last "n_bias" elements
    w1 = tmp_weights[:vars_ * nodes[0]] 
    w11 = np.reshape(w1, (vars_, nodes[0])) # w11 = w1 but with new dimensions

    a_sig = 1 / (1 + np.exp(-( np.dot(tmp_xx, w11) + np.ones((obs, nodes[0])) * w_bias[:nodes[0]] ))) # 1st layer

    ind = len(w1)

    for ii in range(1, len(nodes)): # Going through hidden layers
        w1 = tmp_weights[ ind: (ind + nodes[ii - 1] * nodes[ii])]
        ind = ind + len(w1)
        w11 = np.reshape(w1, (nodes[ii - 1], nodes[ii]))
        a_sig = 1 / (1 + np.exp(-( np.dot(a_sig, w11) + np.ones((obs, nodes[ii])) * w_bias[np.sum(nodes[:ii - 1]) : np.sum(nodes[:ii])] )))

    w1 = tmp_weights[ind:-n_bias]
    w11 = np.reshape(w1, (nodes[-1], vars_depe))
    out = np.dot(a_sig, w11) + np.ones((obs, vars_depe)) * w_bias[-vars_depe:] # Final output layer

    return out