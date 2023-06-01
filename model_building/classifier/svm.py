from sklearn.svm import SVC


def svm(train_X, train_y, params):
    classifier = SVC(
        C=params['C'],
        gamma=params['gamma'],
        kernel=params['kernel'],
        probability=True,
        random_state=0
    )
    classifier.fit(train_X, train_y)
    return classifier
