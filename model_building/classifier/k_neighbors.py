from sklearn.neighbors import KNeighborsClassifier


def k_neighbors(train_X, train_y, params):
    classifier = KNeighborsClassifier(
        n_neighbors=params['n_neighbors'],
        weights=params['weights']
    )
    classifier.fit(train_X, train_y)
    return classifier
