import pandas as pd
import functions
import models
import settings
import variables


def actLearningTest():

    # Read Treated Data from CSV
    treatedData = pd.read_csv(settings.treatedLinksPath)
    treatedData = treatedData[treatedData['y'].notnull()]

    # Get the new active learning labeled data
    actLearnData = pd.read_csv(settings.actLearningExamplesPath)
    actLearnData = actLearnData[actLearnData['y'].notnull()]
    actLearnData['new'] = 1

    # Concatenate the 2 datasets
    data = pd.concat([treatedData, actLearnData.drop('p', axis=1)])

    # Treat the concatenated data
    data = functions.treat(data)

    # Clean and create features
    cleanedData = functions.clean(data)
    cleanedData['new'] = data['new'].fillna(0)
    features = functions.createFeatures(cleanedData)
    y = data['y'].copy()

    # Get train and test masks
    maskTrainTest = settings.getMaskTrainTest(cleanedData)

    # Test: increase both datasets and run the first model
    variables.mlData['xTrain'], variables.mlData['xTest'] = features[maskTrainTest['maskTrain']], features[maskTrainTest['maskTest']]
    variables.mlData['yTrain'], variables.mlData['yTest'] = y[maskTrainTest['maskTrain']], y[maskTrainTest['maskTest']]

    # Extract data from text
    variables.mlData = functions.dataFromText(cleanedData, maskTrainTest['maskTrain'], maskTrainTest['maskTest'], variables.mlData, settings.tfidfParameters)

    # Use the model and look at the scores
    variables.mlData = models.randomForestWMetrics(variables.mlData)

    # store the cleaned data and features
    variables.mlData['cleanedData'] = cleanedData
    variables.mlData['features'] = features
