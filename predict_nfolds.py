from datetime import datetime

def predict_nfolds(a_all_all, a_layer1_all, n_per_part_all, inds_all_all, x_new, i_new, vars_, cc1, xx_fold_all, nof_folds, maetrs, neurons_all):

    predl1_all = []

    for i in range(nof_folds):
        from datetime import datetime
        print(datetime.now().strftime("%H:%M:%S"), f" Predicting {i + 1} of {nof_folds}")

        a_all_tmp = a_all_all[i]
        a_layer1_tmp = a_layer1_all[i]
        n_per_part_tmp = n_per_part_all[i]
        inds_all_tmp = inds_all_all[i]
        xx_fold_tmp = xx_fold_all[i]

        predl1_new, layer_new = predict_new_rbf(a_all_tmp, a_layer1_tmp, x_new, i_new, neurons_all[i], n_per_part_tmp, inds_all_tmp, xx_fold_tmp, vars_, cc1) # type: ignore

        predl1_all = predl1_all.append(predl1_new)

    return predl1_all 