import numpy as np
from sklearn import metrics

test_set_src_path = "./data_preprocessing/output/test.csv"
test_X_src_path = "./feature_selection/output/test_X.csv"
classifier_src_dir = "./model_building/output"

predict_y_dst_dir = "./label_prediction/output"
evaluation_result_dst_dir = "./label_prediction/evaluation"

random_forest_params = {
    'n_estimators': 200,
    'max_depth': None
}

k_neighbors_params = {
    'n_neighbors': 10,
    'weights': 'distance'
}

xgboost_params = {
    'learning_rate': 0.02,
    'n_estimators': 100,
    'reg_lambda': 1
}

classifier_params_map = {
    'random_forest': random_forest_params,
    'k_neighbors': k_neighbors_params,
    'xgboost': xgboost_params
}


def cal_metric(test_y, predict_y, pro_predict_y=None):
    ret = dict()
    cm = metrics.confusion_matrix(y_true=test_y, y_pred=predict_y)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    ret['confusion_matrix'] = [[int(TN), int(FP)], [int(FN), int(TP)]]
    if pro_predict_y is not None:
        FPR, TPR, thresholds = metrics.roc_curve(test_y, pro_predict_y[:, 1], pos_label=1)
        ret['auc'] = metrics.auc(FPR, TPR)
    ret['sensitivity'] = TP / (TP + FN)
    ret['specification'] = TN / (FP + TN)
    ret['g-mean'] = np.sqrt(ret['sensitivity'] * ret['specification'])
    return ret
