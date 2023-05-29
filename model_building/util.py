from classifier import *

train_X_src_path = "./sample_balance/output/train_X.csv"
train_y_src_path = "./sample_balance/output/train_y.csv"

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

svm_params_lst = [
    {
        'C': 1,
        'gamma': 1,
        'kernel': 'rbf'
    }
]

classifier_map = {
    'random_forest': (random_forest, random_forest_params_lst),
    'k_neighbors': (k_neighbors, k_neighbors_params_lst),
    'xgboost': (xgboost, xgboost_params_lst),
    'svm': (svm, svm_params_lst)
}
