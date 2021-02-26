import joblib as jb
import models
import optimization
import pandas as pd
import settings
import variables

from sklearn.preprocessing import MaxAbsScaler
from scipy.sparse import csr_matrix
from sklearn.metrics import average_precision_score, roc_auc_score


def mainModel():

    # Random Forest
    variables.mlData = models.randomForestWMetrics(variables.mlData)

    # Optimized LGBM
    variables.mlData['paramsOptimizedLGBM'], variables.mlData['apsOptimizedLGBM'] = optimization.optimizeLGBM(settings.LGBMSpace)

    # Scaling
    variables.mlData['scaledXTrain'] = csr_matrix(variables.mlData['xTrain'].copy())
    variables.mlData['scaledXTest'] = csr_matrix(variables.mlData['xTest'].copy())

    scaler = MaxAbsScaler()
    variables.mlData['scaledXTrain'] = scaler.fit_transform(variables.mlData['scaledXTrain'])
    variables.mlData['scaledXTest'] = scaler.transform(variables.mlData['scaledXTest'])

    # Logistic Regression
    variables.mlData = models.logisticRegressionWMetrics(variables.mlData)

    # Testing the correlation between models
    pd.DataFrame({'RF': variables.mlData['probRF'], 'LBGM': variables.mlData['probLGBM'], 'LR': variables.mlData['probLR']}).corr()

    # Final step: ensembling everything
    p = (variables.mlData['probRF'] + variables.mlData['probLGBM'] + variables.mlData['probLR']) / 3

    # Metrics for testing the ensemble
    aps = average_precision_score(variables.mlData['yTest'], p)
    roc_auc = roc_auc_score(variables.mlData['yTest'], p)

    # Save the models on disk
    jb.dump(variables.mlData['modelRF'], settings.RandomForestPath)
    jb.dump(variables.mlData['modelLGBM'], settings.lightGBMPath)
    jb.dump(variables.mlData['modelLR'], settings.logisticRegressionPath)
    jb.dump(variables.mlData['tFidVec'], settings.VectorizerPath)
