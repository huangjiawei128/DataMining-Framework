from classifier import *

train_X_src_path = "./sample_balance/output/train_X.csv"
train_y_src_path = "./sample_balance/output/train_y.csv"

classifier_dst_dir = "./model_building/output"

classifier_map = {
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

classifier_params_map = {
    'random_forest': random_forest_params,
    'k_neighbors': k_neighbors_params,
    'xgboost': xgboost_params
}
