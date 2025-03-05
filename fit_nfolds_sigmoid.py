import numpy as np
from sklearn.model_selection import train_test_split # type: ignore
from datetime import datetime

def fit_nfolds_sigmoid(xx_train, yy_train, nof_folds, vars, i_train, i_fold, neurons):

    maetrs = []
    a_all_all = []
    a_layer1_all = []
    n_per_part_all = []
    inds_all_all = []
    xx_fold_all = []
    neurons_all = []

    for i in range(nof_folds):

        inds_fold = np.random.permutation(i_train)[:i_fold]
        xx_fold = xx_train[inds_fold, :]
        yy_fold = yy_train[inds_fold]
        neurons_all.append(neurons)

        inds_all_vec_unvec, n_per_part_new = ___clustering(neurons, xx_fold, 300) # type: ignore
        a_all, a_layer1, layer1 = train_layer_1_sigmoid(neurons, vars, i_fold, n_per_part_new, inds_all_vec_unvec, xx_fold, yy_fold) # type: ignore

        a_all_all.append(a_all)
        a_layer1_all.append(a_layer1)
        n_per_part_all.append(n_per_part_new)
        inds_all_all.append(inds_all_vec_unvec)

        predl1 = np.dot(np.hstack([layer1, np.ones((i_fold, 1))]), a_layer1)

        maetr = np.mean(np.abs(yy_fold - predl1))
        maetrs.append(maetr)

        print(f"{datetime.now().strftime('%H:%M:%S')} {i} maetr={maetr}")
    
    return maetrs, a_all_all, a_layer1_all, n_per_part_all, inds_all_all, xx_fold_all, neurons_all