from sklearn.ensemble import RandomForestRegressor


def random_forest(train_X, train_y, params):
    regressor = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=0
    )
    regressor.fit(train_X, train_y)
    return regressor
