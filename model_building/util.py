from classifier import *

train_X_src_path = "./feature_selection/output/train_X.csv"
train_X_cp_src_path = "./feature_selection/output/train_X_cp.csv"
train_y_src_path = "./data_preprocessing/output/train_y.csv"

classifier_dst_dir = "./model_building/output"

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
        'func': random_forest,
        'params_lst': random_forest_params_lst
    },
    'k_neighbors': {
        'need_cp_samples': True,
        'func': k_neighbors,
        'params_lst': k_neighbors_params_lst
    },
    'xgboost': {
        'need_cp_samples': True,
        'func': xgboost,
        'params_lst': xgboost_params_lst
    },
}
