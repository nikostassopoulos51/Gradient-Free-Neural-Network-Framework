import numpy as np

def train_layer_1_rbf_pde(neurons, vars, obs, n_per_part, inds_all, x, cc, xb, yb, y_source):
    a_all = []  
    ooops = 0

    for i in range(neurons):
        if n_per_part[i + 1] - n_per_part[i] != 0:
            try:
                # Randomly take data for this neuron
                x1 = x[inds_all[n_per_part[i] + 1 : n_per_part[i + 1]], :]
                obs1 = x1.shape[0]

                # Calculate RBF derivatives for x & y
                phi_x = calc_phi_deriv_deriv(x1, x1, obs1, obs1, cc, vars, 0)  # var_seq = 0
                phi_y = calc_phi_deriv_deriv(x1, x1, obs1, obs1, cc, vars, 1)  # var_seq = 1
            
                # RBF kernel
                phi = calc_phi(x1, xb, obs1, xb.shape[0], cc, vars) 
            
                # Solve for weights using least squares
                A = np.vstack([(phi_x + phi_y), phi]) 
                b = np.hstack([y_source[inds_all[n_per_part[i] + 1 : n_per_part[i + 1]]], yb])
                atmp = np.linalg.lstsq(A.T, b, rcond = None)[0]
            
                if not np.isnan(atmp).any() and not np.isinf(atmp).any():
                    a_all.append(atmp)
                else:
                    ooops += 1

            except Exception as ex:
                ooops += 1
                print(f"{i} {ex}")
                break