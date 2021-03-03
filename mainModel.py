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

    # Optimize a LGBM to get ideal parameters for TFidVec and models
    variables.mlData['paramsOptimizedLGBM'], variables.mlData['apsOptimizedLGBM'] = optimization.optimizeLGBM(settings.LGBMSpace)

    # Make the main TFiDVectorizer with the params of the optimization
    maskTrainTest = settings.getMaskTrainTest(variables.mlData['cleanedData'])
    variables.mlData = functions.dataFromText(variables.mlData['cleanedData'], maskTrainTest['maskTrain'], maskTrainTest['maskTest'],
                                              variables.mlData,
                                              {'min_df': variables.mlData['paramsOptimizedLGBM'][6], 'ngram_range': (1, variables.mlData['paramsOptimizedLGBM'][7])})

    # Make the main LGBM model from the ideal parameters
    variables.mlData = models.lgbmWMetrics(variables.mlData, 2 ** variables.mlData['paramsOptimizedLGBM'][1],
                                           variables.mlData['paramsOptimizedLGBM'][0],
                                           variables.mlData['paramsOptimizedLGBM'][1],
                                           variables.mlData['paramsOptimizedLGBM'][2],
                                           variables.mlData['paramsOptimizedLGBM'][3],
                                           variables.mlData['paramsOptimizedLGBM'][4],
                                           variables.mlData['paramsOptimizedLGBM'][5])

    # Random Forest
    variables.mlData = models.randomForestWMetrics(variables.mlData)

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
