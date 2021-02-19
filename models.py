from sklearn.ensemble import RandomForestClassifier
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


def lgbmWMetrics(mlData, lgbmParameters):

    modelLGBM = LGBMClassifier(lgbmParameters, num_leaves=2 ** lgbmParameters['max_depth'], random_state=0, class_weight="balanced", n_jobs=6)
    modelLGBM.fit(mlData['xTrain'], mlData['yTrain'])
    mlData['modelLGBM'] = modelLGBM

    mlData['probLGBM'] = modelLGBM.predict_proba(mlData['xTest'])[:, 1]

    # Metrics for testing the model
    mlData['apsLGBM'] = average_precision_score(mlData['yTest'], mlData['probLGBM'])
    mlData['roc_aucLGBM'] = roc_auc_score(mlData['yTest'], mlData['probLGBM'])

    return mlData
