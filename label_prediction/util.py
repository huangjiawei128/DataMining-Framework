import numpy as np
from sklearn import metrics

test_set_src_path = "./data_preprocessing/output/test.csv"
test_X_src_path = "./feature_selection/output/test_X.csv"
classifier_src_dir = "./model_building/output"

predict_y_dst_dir = "./label_prediction/output"
evaluation_result_dst_dir = "./label_prediction/evaluation"

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

svm_params_lst = [
    {
        'C': 1,
        'gamma': 1,
        'kernel': 'rbf'
    }
]

classifier_params_lst_map = {
    'random_forest': random_forest_params_lst,
    'k_neighbors': k_neighbors_params_lst,
    'xgboost': xgboost_params_lst,
    'svm': svm_params_lst
}


def cal_metric(test_y, predict_y, pro_predict_y=None):
    ret = dict()
    cm = metrics.confusion_matrix(y_true=test_y, y_pred=predict_y)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    ret['confusion_matrix'] = [[int(TN), int(FP)], [int(FN), int(TP)]]
    if pro_predict_y is not None:
        FPR, TPR, thresholds = metrics.roc_curve(test_y, pro_predict_y[:, 1], pos_label=1)
        ret['auc'] = metrics.auc(FPR, TPR)
    ret['precision'] = metrics.precision_score(y_true=test_y, y_pred=predict_y)
    ret['recall'] = metrics.recall_score(y_true=test_y, y_pred=predict_y)
    ret['f1_score'] = metrics.f1_score(y_true=test_y, y_pred=predict_y)
    ret['sensitivity'] = TP / (TP + FN)
    ret['specification'] = TN / (FP + TN)
    ret['g-mean'] = np.sqrt(ret['sensitivity'] * ret['specification'])
    return ret
