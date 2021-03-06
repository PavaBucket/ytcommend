from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


def randomForestWMetrics(xTrain, yTrain, xTest, yTest):

    # Modelling: Random Forest
    model = RandomForestClassifier(n_estimators=1000, random_state=0, min_samples_leaf=2, class_weight="balanced", n_jobs=6)
    model.fit(xTrain, yTrain)

    prob = model.predict_proba(xTest)[:, 1]

    # Metrics for testing the model
    aps = average_precision_score(yTest, prob)
    roc_auc = roc_auc_score(yTest, prob)

    return model, prob, aps, roc_auc


def decisionTreeWMetrics(xTrain, yTrain, xTest, yTest):

    # Modelling: Decision Tree
    model = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight="balanced")
    model.fit(xTrain, yTrain)

    prob = model.predict_proba(xTest)[:, 1]

    # Metrics for testing the model
    aps = average_precision_score(yTest, prob)
    roc_auc = roc_auc_score(yTest, prob)

    return model, prob, aps, roc_auc


def lgbmWMetrics(xTrain, yTrain, xTest, yTest, num_leaves, learning_rate, max_depth, min_child_samples, subsample, colsample_bytree, n_estimators):

    model = LGBMClassifier(num_leaves=num_leaves, random_state=0, class_weight="balanced", n_jobs=6, learning_rate=learning_rate,
                           max_depth=max_depth, min_child_samples=min_child_samples, subsample=subsample, colsample_bytree=colsample_bytree,
                           n_estimators=n_estimators)
    model.fit(xTrain, yTrain)

    prob = model.predict_proba(xTest)[:, 1]

    # Metrics for testing the model
    aps = average_precision_score(yTest, prob)
    roc_auc = roc_auc_score(yTest, prob)

    return model, prob, aps, roc_auc


def logisticRegressionWMetrics(scaledXTrain, yTrain, scaledXTest, yTest):

    model = LogisticRegression(C=0.5, n_jobs=6, random_state=0)

    model.fit(scaledXTrain, yTrain)

    prob = model.predict_proba(scaledXTest)[:, 1]

    # Metrics for testing the model
    aps = average_precision_score(yTest, prob)
    roc_auc = roc_auc_score(yTest, prob)

    return model, prob, aps, roc_auc
