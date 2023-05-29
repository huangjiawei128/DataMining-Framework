from sklearn.ensemble import RandomForestClassifier


def random_forest(train_X, train_y, params):
    classifier = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=0
    )
    classifier.fit(train_X, train_y)
    return classifier
