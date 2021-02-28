from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier


def randomForestWMetrics(mlData):

    # Modelling: Random Forest
    modelRF = RandomForestClassifier(n_estimators=1000, random_state=0, min_samples_leaf=2, class_weight="balanced", n_jobs=6)
    modelRF.fit(mlData['xTrain'], mlData['yTrain'])
    mlData['modelRF'] = modelRF

    mlData['probRF'] = modelRF.predict_proba(mlData['xTest'])[:, 1]

    # Metrics for testing the model
    mlData['apsRF'] = average_precision_score(mlData['yTest'], mlData['probRF'])
    mlData['roc_aucRF'] = roc_auc_score(mlData['yTest'], mlData['probRF'])

    return mlData


def decisionTreeWMetrics(mlData):

    # Modelling: Decision Tree
    modelDT = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight="balanced")
    modelDT.fit(mlData['xTrain'], mlData['yTrain'])
    mlData['modelDT'] = modelDT

    mlData['probDT'] = modelDT.predict_proba(mlData['xTest'])[:, 1]

    # Metrics for testing the model
    mlData['apsDT'] = average_precision_score(mlData['yTest'], mlData['probDT'])
    mlData['roc_aucDT'] = roc_auc_score(mlData['yTest'], mlData['probDT'])

    return mlData


def lgbmWMetrics(mlData, num_leaves, learning_rate, max_depth, min_child_samples, subsample, colsample_bytree, n_estimators):

    modelLGBM = LGBMClassifier(num_leaves=num_leaves, random_state=0, class_weight="balanced", n_jobs=6, learning_rate=learning_rate,
                               max_depth=max_depth, min_child_samples=min_child_samples, subsample=subsample, colsample_bytree=colsample_bytree,
                               n_estimators=n_estimators)
    modelLGBM.fit(mlData['xTrain'], mlData['yTrain'])
    mlData['modelLGBM'] = modelLGBM

    mlData['probLGBM'] = modelLGBM.predict_proba(mlData['xTest'])[:, 1]

    # Metrics for testing the model
    mlData['apsLGBM'] = average_precision_score(mlData['yTest'], mlData['probLGBM'])
    mlData['roc_aucLGBM'] = roc_auc_score(mlData['yTest'], mlData['probLGBM'])

    return mlData


def logisticRegressionWMetrics(mlData):

    modelLR = LogisticRegression(C=0.5, n_jobs=6, random_state=0)
    modelLR.fit(mlData['scaledXTrain'], mlData['yTrain'])
    mlData['modelLR'] = modelLR

    mlData['probLR'] = modelLR.predict_proba(mlData['scaledXTest'])[:, 1]

    # Metrics for testing the model
    mlData['apsLR'] = average_precision_score(mlData['yTest'], mlData['probLR'])
    mlData['roc_aucLR'] = roc_auc_score(mlData['yTest'], mlData['probLR'])

    return mlData
