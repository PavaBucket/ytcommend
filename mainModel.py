import functions
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

    # Optimize to get ideal parameters for LGBM model
    variables.mlData['paramsOptimizedLGBM'], variables.mlData['apsOptimizedLGBM'] = optimization.optimizeLGBM(settings.LGBMSpace)

    # Make the main LGBM model from the optimized parameters and the settings TFid
    variables.mlData['modelLGBM'], variables.mlData['probLGBM'], variables.mlData['apsLGBM'], variables.mlData['roc_aucLGBM'] = models.lgbmWMetrics(
        variables.mlData['xTrain'], variables.mlData['yTrain'], variables.mlData['xTest'], variables.mlData['yTest'],
        2 ** variables.mlData['paramsOptimizedLGBM'][1],
        variables.mlData['paramsOptimizedLGBM'][0],
        variables.mlData['paramsOptimizedLGBM'][1],
        variables.mlData['paramsOptimizedLGBM'][2],
        variables.mlData['paramsOptimizedLGBM'][3],
        variables.mlData['paramsOptimizedLGBM'][4],
        variables.mlData['paramsOptimizedLGBM'][5])

    # Random Forest
    variables.mlData['modelRF'], variables.mlData['probRF'], variables.mlData['apsRF'], variables.mlData['roc_aucRF'] = models.randomForestWMetrics(variables.mlData['xTrain'], variables.mlData['yTrain'], variables.mlData['xTest'], variables.mlData['yTest'])

    # Scaling
    variables.mlData['scaledXTrain'] = csr_matrix(variables.mlData['xTrain'].copy())
    variables.mlData['scaledXTest'] = csr_matrix(variables.mlData['xTest'].copy())

    scaler = MaxAbsScaler()
    variables.mlData['scaledXTrain'] = scaler.fit_transform(variables.mlData['scaledXTrain'])
    variables.mlData['scaledXTest'] = scaler.transform(variables.mlData['scaledXTest'])

    # Logistic Regression
    variables.mlData['modelLR'], variables.mlData['probLR'], variables.mlData['apsLR'], variables.mlData['roc_aucLR'] = models.logisticRegressionWMetrics(variables.mlData['scaledXTrain'], variables.mlData['yTrain'], variables.mlData['scaledXTest'], variables.mlData['yTest'])

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
