from skopt import forest_minimize
import functions
import models
import settings
import variables


def LGBMTargetFunction(parameters):

    learning_rate = parameters[0]
    max_depth = parameters[1]
    min_child_samples = parameters[2]
    subsample = parameters[3]
    colsample_bytree = parameters[4]
    n_estimators = parameters[5]
    min_df = parameters[6]
    ngram_range = (1, parameters[7])

    print(parameters)

    # Get train and test masks
    maskTrainTest = settings.getMaskTrainTest(variables.mlData['cleanedData'])

    # Extract text
    variables.mlData = functions.dataFromText(variables.mlData['cleanedData'], maskTrainTest['maskTrain'], maskTrainTest['maskTest'], variables.mlData, {'min_df': min_df, 'ngram_range': ngram_range})

    # Run model
    variables.mlData = models.lgbmWMetrics(variables.mlData, 2 ** max_depth, learning_rate, max_depth, min_child_samples, subsample, colsample_bytree, n_estimators)

    print(variables.mlData['apsLGBM'])
    print(variables.mlData['roc_aucLGBM'])

    return -variables.mlData['apsLGBM']


def optimizeLGBM(space):

    result = forest_minimize(LGBMTargetFunction, space, random_state=160745, n_random_starts=20, n_calls=50, verbose=1)

    return (result.x, result.fun)
