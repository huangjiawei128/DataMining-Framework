import numpy as np
from sklearn import metrics

test_set_src_path = "./data_preprocessing/output/test.csv"
test_X_src_path = "./feature_selection/output/test_X.csv"
regressor_src_dir = "./model_building/output"

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

regressor_params_map = {
    'random_forest': random_forest_params,
    'k_neighbors': k_neighbors_params,
    'xgboost': xgboost_params
}


def cal_metric(test_y, predict_y):
    ret = dict()
    ret['R2'] = metrics.r2_score(y_true=test_y, y_pred=predict_y)
    ret['MSE'] = metrics.mean_squared_error(y_true=test_y, y_pred=predict_y)
    ret['RMSE'] = np.sqrt(ret['MSE'])
    ret['MAE'] = metrics.mean_absolute_error(y_true=test_y, y_pred=predict_y)
    ret['MAPE'] = metrics.mean_absolute_percentage_error(y_true=test_y, y_pred=predict_y)
    return ret
