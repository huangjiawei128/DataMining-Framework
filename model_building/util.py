from regressor import *
from sklearn import metrics

train_set_src_path = "./data_preprocessing/output/train.csv"
train_X_src_path = "./feature_selection/output/train_X.csv"

regressor_dst_dir = "./model_building/output"

regressor_map = {
    'random_forest': random_forest,
    'k_neighbors': k_neighbors,
    'xgboost': xgboost
}

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
    'n_estimators': 200,
    'reg_lambda': 0.1
}

regressor_params_map = {
    'random_forest': random_forest_params,
    'k_neighbors': k_neighbors_params,
    'xgboost': xgboost_params
}
