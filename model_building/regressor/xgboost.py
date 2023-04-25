from xgboost import XGBRegressor


def xgboost(train_X, train_y, params):
    regressor = XGBRegressor(
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        random_state=0
    )
    regressor.fit(train_X, train_y)
    return regressor
