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

    # Get train and test masks and segmentations
    maskTrainTest = settings.getMaskTrainTest(variables.mlData['cleanedData'])
    xTrain, xTest = variables.mlData['features'][maskTrainTest['maskTrain']], variables.mlData['features'][maskTrainTest['maskTest']]
    yTrain, yTest = variables.mlData['y'][maskTrainTest['maskTrain']], variables.mlData['y'][maskTrainTest['maskTest']]

    # Extract data from text
    tFidVec, titleBOWTrain, titleBOWTest = functions.dataFromText(variables.mlData['cleanedData']['title'], maskTrainTest['maskTrain'], maskTrainTest['maskTest'], {'min_df': min_df, 'ngram_range': ngram_range})

    # Include extracted text into training and testing
    xTrain = functions.mergeDataFrames(xTrain, titleBOWTrain)
    xTest = functions.mergeDataFrames(xTest, titleBOWTest)

    # Run model
    model, prob, aps, roc_auc = models.lgbmWMetrics(xTrain, yTrain, xTest, yTest, 2 ** max_depth, learning_rate, max_depth, min_child_samples, subsample, colsample_bytree, n_estimators)

    print(aps)
    print(roc_auc)

    return -aps


def optimizeLGBM(space):

    result = forest_minimize(LGBMTargetFunction, space, random_state=160745, n_random_starts=20, n_calls=50, verbose=1)

    return (result.x, result.fun)
