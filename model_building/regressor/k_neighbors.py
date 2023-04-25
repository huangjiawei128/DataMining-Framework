from sklearn.neighbors import KNeighborsRegressor


def k_neighbors(train_X, train_y, params):
    regressor = KNeighborsRegressor(
        n_neighbors=params['n_neighbors'],
        weights=params['weights']
    )
    regressor.fit(train_X, train_y)
    return regressor
