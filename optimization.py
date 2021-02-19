from skopt import forest_minimize
import functions
import models
import settings


def optimizeLGBM(mlData, modelParameters, tfidfParameters):

    print(modelParameters)

    # Get train and test masks
    maskTrainTest = settings.getMaskTrainTest(mlData['cleanedData'])

    # Extracting text
    functions.dataFromText(mlData['cleanedData'], maskTrainTest['maskTrain'], maskTrainTest['maskTest'], mlData, tfidfParameters)

    models.lgbmWMetrics(mlData, modelParameters)

    print(mlData['roc_aucLGBM'])

    return -mlData['apsLGBM']
