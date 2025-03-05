import numpy as np 

def calc_phi(x, xi, n, nn, c, vars):
    
    # x and xi are 2D numpy arrays with shapes (n, vars) and (nn, vars), respectively.

    phi = np.zeros((nn, n)) # Initialize

    # Calculate the distance between i-th element of matrix xi & the j-th element of x - Sum the squared differences for each feature k
    
    for i in range(nn):
        for j in range(n):
            dr = 0
            for k in range(vars):
                dr += (xi[i, k] - x[j, k]) ** 2
            phi[i, j] = np.exp(-dr / c) # Exponential
    
    return phi