import numpy as np
from datetime import datetime

def fit_nfolds(xx_train, yy_train, nof_folds, vars, i_train, i_fold, cc1, neurons):

    maetrs = []
    a_all_all = []
    a_layer1_all = []
    n_per_part_all = []
    inds_all_all = []
    xx_fold_all = []
    neurons_all = []
    pred_folds_all = np.zeros((i_fold, nof_folds))

    for i in range(nof_folds):
        
        inds_fold = np.random.permutation(i_train)[:i_fold] # Randomly shuffle the indices from 0 --> i_train - 1 and then select the 1st i_fold indice
        xx_fold = xx_train[inds_fold, :] # xx_fold contains the rows of xx_train which correspond to the indices of inds_fold
        yy_fold = yy_train[inds_fold] # yy_fold contains the elements of yy_train which correspond to the indices of inds_fold
        xx_fold_all.append(xx_fold) # Adds xx_fold to xx_fold_all

        a_all, a_layer1, layer1 = train_layer_1_rbf(neurons, vars, i_fold, n_per_part_new, inds_all_vec_unvec, xx_fold, yy_fold, cc1) # type: ignore
        
        a_all_all.append(a_all)  # Store neuron weights
        a_layer1_all.append(a_layer1)  # Store output layer weights

        predl1 = np.hstack([layer1, np.ones((i_fold, 1))]) @ a_layer1
        pred_folds_all[:, i] = predl1  # Store predictions

        maetr = np.mean(np.abs(yy_fold - predl1))
        maetrs.append(maetr)
        '''
        Find the Mean Absolute Error:
        --> Substract predicted values from actual values
        --> Take the absolute of the substraction
        --> Find average absolute error
        '''

        print(f"{datetime.now().strftime('%H:%M:%S')} Fold {i+1}, MAE: {maetr}")
    
    return maetrs, a_all_all, a_layer1_all, n_per_part_all, inds_all_all, xx_fold_all, neurons_all, pred_folds_all