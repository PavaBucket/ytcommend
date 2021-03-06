import models
import variables


def firstModel():

    variables.mlData['modelRF'], variables.mlData['probRF'], variables.mlData['apsRF'], variables.mlData['roc_aucRF'] = models.randomForestWMetrics(variables.mlData['xTrain'], variables.mlData['yTrain'], variables.mlData['xTest'], variables.mlData['yTest'])
