import numpy as np 

def calc_phi_deriv_deriv(x, xi, n, nn, c, vars, var_seq):
    
    phi_der_der = np.zeros((nn, n)) # Initialize


    for i in range(nn):
        for j in range(n):
            dr = 0
            for k in range(vars):
                dr += (xi[i, k] - x[j, k]) ** 2
            r = dr
    
    phi_der_der[i, j] = ( (-2 * np.exp(-r / c) / c) + ( (-2 * xi[i, var_seq] + 2 * x[j, var_seq]) ** 2) ) * np.exp(-r / c) / (c ** 2)

    return phi_der_der