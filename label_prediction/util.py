<<<<<<< HEAD
import numpy as np
from sklearn import metrics

test_X_src_path = "./feature_selection/output/test_X.csv"
test_X_cp_src_path = "./feature_selection/output/test_X_cp.csv"
test_y_src_path = "./data_preprocessing/output/test_y.csv"
classifier_src_dir = "./model_building/output"

predict_y_dst_dir = "./label_prediction/output"
evaluation_result_dst_dir = "./label_prediction/evaluation"
roc_curve_dst_dir = "./label_prediction/roc_curve"

random_forest_params_lst = [
    {
        'n_estimators': 50,
        'max_depth': 10
    },
    {
        'n_estimators': 50,
        'max_depth': 20
    },
    {
        'n_estimators': 50,
        'max_depth': None
    },
    {
        'n_estimators': 100,
        'max_depth': 10
    },
    {
        'n_estimators': 100,
        'max_depth': 20
    },
    {
        'n_estimators': 100,
        'max_depth': None
    },
    {
        'n_estimators': 200,
        'max_depth': 10
    },
    {
        'n_estimators': 200,
        'max_depth': 20
    },
    {
        'n_estimators': 200,
        'max_depth': None
    }
]

k_neighbors_params_lst = [
    {
        'n_neighbors': 10,
        'weights': 'uniform'
    },
    {
        'n_neighbors': 10,
        'weights': 'distance'
    },
    {
        'n_neighbors': 20,
        'weights': 'uniform'
    },
    {
        'n_neighbors': 20,
        'weights': 'distance'
    },
    {
        'n_neighbors': 40,
        'weights': 'uniform'
    },
    {
        'n_neighbors': 40,
        'weights': 'distance'
    }
]

xgboost_params_lst = [
    {
        'learning_rate': 0.02,
        'n_estimators': 100,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.02,
        'n_estimators': 100,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.02,
        'n_estimators': 200,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.02,
        'n_estimators': 200,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 100,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 100,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 200,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 200,
        'reg_lambda': 1
    }
]

classifier_map = {
    'random_forest': {
        'need_cp_samples': True,
        'params_lst': random_forest_params_lst
    },
    'k_neighbors': {
        'need_cp_samples': True,
        'params_lst': k_neighbors_params_lst
    },
    'xgboost': {
        'need_cp_samples': True,
        'params_lst': xgboost_params_lst
    },
}


def cal_metric(test_y, predict_y, pro_predict_y=None):
    ret = dict()
    cm = metrics.confusion_matrix(y_true=test_y, y_pred=predict_y)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    ret['confusion_matrix'] = [[int(TN), int(FP)], [int(FN), int(TP)]]
    if pro_predict_y is not None:
        FPR, TPR, thresholds = metrics.roc_curve(test_y, pro_predict_y[:, 1], pos_label=1)
        ret['FPR'], ret['TPR'] = FPR, TPR
        ret['auc'] = metrics.auc(FPR, TPR)
    ret['precision'] = metrics.precision_score(y_true=test_y, y_pred=predict_y)
    ret['recall'] = metrics.recall_score(y_true=test_y, y_pred=predict_y)
    ret['f1_score'] = metrics.f1_score(y_true=test_y, y_pred=predict_y)
    ret['sensitivity'] = TP / (TP + FN)
    ret['specification'] = TN / (FP + TN)
    ret['g-mean'] = np.sqrt(ret['sensitivity'] * ret['specification'])
    return ret
=======
import numpy as np
from sklearn import metrics

test_X_src_path = "./feature_selection/output/test_X.csv"
test_X_cp_src_path = "./feature_selection/output/test_X_cp.csv"
test_y_src_path = "./data_preprocessing/output/test_y.csv"
classifier_src_dir = "./model_building/output"

predict_y_dst_dir = "./label_prediction/output"
evaluation_result_dst_dir = "./label_prediction/evaluation"
roc_curve_dst_dir = "./label_prediction/roc_curve"

random_forest_params_lst = [
    {
        'n_estimators': 50,
        'max_depth': 10
    },
    {
        'n_estimators': 50,
        'max_depth': 20
    },
    {
        'n_estimators': 50,
        'max_depth': None
    },
    {
        'n_estimators': 100,
        'max_depth': 10
    },
    {
        'n_estimators': 100,
        'max_depth': 20
    },
    {
        'n_estimators': 100,
        'max_depth': None
    },
    {
        'n_estimators': 200,
        'max_depth': 10
    },
    {
        'n_estimators': 200,
        'max_depth': 20
    },
    {
        'n_estimators': 200,
        'max_depth': None
    }
]

k_neighbors_params_lst = [
    {
        'n_neighbors': 10,
        'weights': 'uniform'
    },
    {
        'n_neighbors': 10,
        'weights': 'distance'
    },
    {
        'n_neighbors': 20,
        'weights': 'uniform'
    },
    {
        'n_neighbors': 20,
        'weights': 'distance'
    },
    {
        'n_neighbors': 40,
        'weights': 'uniform'
    },
    {
        'n_neighbors': 40,
        'weights': 'distance'
    }
]

xgboost_params_lst = [
    {
        'learning_rate': 0.02,
        'n_estimators': 100,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.02,
        'n_estimators': 100,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.02,
        'n_estimators': 200,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.02,
        'n_estimators': 200,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 100,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 100,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 100,
        'reg_lambda': 1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 200,
        'reg_lambda': 0.1
    },
    {
        'learning_rate': 0.5,
        'n_estimators': 200,
        'reg_lambda': 1
    }
]

classifier_map = {
    'random_forest': {
        'need_cp_samples': True,
        'params_lst': random_forest_params_lst
    },
    'k_neighbors': {
        'need_cp_samples': True,
        'params_lst': k_neighbors_params_lst
    },
    'xgboost': {
        'need_cp_samples': True,
        'params_lst': xgboost_params_lst
    },
}


def cal_metric(test_y, predict_y, pro_predict_y=None):
    ret = dict()
    cm = metrics.confusion_matrix(y_true=test_y, y_pred=predict_y)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    ret['confusion_matrix'] = [[int(TN), int(FP)], [int(FN), int(TP)]]
    if pro_predict_y is not None:
        FPR, TPR, thresholds = metrics.roc_curve(test_y, pro_predict_y[:, 1], pos_label=1)
        ret['FPR'], ret['TPR'] = FPR, TPR
        ret['auc'] = metrics.auc(FPR, TPR)
    ret['precision'] = metrics.precision_score(y_true=test_y, y_pred=predict_y)
    ret['recall'] = metrics.recall_score(y_true=test_y, y_pred=predict_y)
    ret['f1_score'] = metrics.f1_score(y_true=test_y, y_pred=predict_y)
    ret['sensitivity'] = TP / (TP + FN)
    ret['specification'] = TN / (FP + TN)
    ret['g-mean'] = np.sqrt(ret['sensitivity'] * ret['specification'])
    return ret
>>>>>>> 2a6ebe51ab223003d660b142310d9fe2ef72b085
