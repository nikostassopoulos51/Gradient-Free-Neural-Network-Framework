from datetime import datetime

def predict_nfolds_sigmoid(a_all_all, a_layer1_all, n_per_part_all, inds_all_all, x_new, i_new, vars_, xx_fold_all, nof_folds, maetrs, neurons_all):
    
    predl1_all = []
    
    for i in range(nof_folds):
        print(datetime.now().strftime("%H:%M:%S"), f" Predicting {i + 1} fold, of {nof_folds} Total.")
        
        a_all_tmp = a_all_all[i]
        a_layer1_tmp = a_layer1_all[i]
        n_per_part_tmp = n_per_part_all[i]
        inds_all_tmp = inds_all_all[i]
        xx_fold_tmp = xx_fold_all[i]
        
        predl1_new, ll = predict_new(a_all_tmp, a_layer1_tmp, x_new, i_new, neurons_all[i]) # type: ignore
        
        predl1_all = predl1_all.append( (predl1_new) * (1 / maetrs[i]) )
    
    return predl1_all
