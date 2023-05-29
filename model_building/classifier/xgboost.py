from xgboost import XGBClassifier


def xgboost(train_X, train_y, params):
    classifier = XGBClassifier(
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        reg_lambda=params['reg_lambda'],
        random_state=0
    )
    classifier.fit(train_X, train_y)
    return classifier
