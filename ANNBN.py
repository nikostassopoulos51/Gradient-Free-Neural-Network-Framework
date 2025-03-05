'''
The ANNBN class implements the architecture for an artificial neural network 
and serves as a container for the training, prediction & clustering functionalities related!!!
'''

import numpy as np 

class ANNBN:
    
    def __init__(self):
        pass

    @staticmethod
    def sigm1(x):
        return 1.0 / (1.0 + np.exp(-x))  # The Sigmoid Activation Function
    
    @staticmethod
    def isigm1(x):
        return -np.log((1.0 - x) / x)  # The Inverse Sigmoid
    
    # Staticmethod because hese methods do not depend on instance variables of the class, 
    # making it clear that they can be called without creating an instance of ANNBN
    
    '''
    ADD ALL THE IMPORTS HERE
    '''

    import train_layer_1_sigmoid , train_layer_1_sigmoid_fast , train_layer_1_sigmoid_fast2 # type: ignore
    import train_layer_1_rbf , train_layer_1_rbf_deriv , train_layer_1_rbf_pde , train_layer_1_rbf_new , train_layer_1_rbf_fast # type: ignore
    import ___clustering , __fast_clustering_sub_sampling # type: ignore
    import calc_phi , calc_phi_deriv , calc_phi_deriv_deriv # type: ignore
    import fit_nfolds , fit_nfolds_sigmoid # type: ignore
    import predict_new , predict_new_rbf , predict_new_rbf_deriv , predict_new_rbf_deriv_deriv # type: ignore
    import predict_nfolds , predict_nfolds_sigmoid # type: ignore
    import deep_nnm  # type: ignore

