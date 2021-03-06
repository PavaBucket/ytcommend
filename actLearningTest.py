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
    xTrain, xTest = features[maskTrainTest['maskTrain']], features[maskTrainTest['maskTest']]
    yTrain, yTest = y[maskTrainTest['maskTrain']], y[maskTrainTest['maskTest']]

    # Extract data from text
    variables.mlData['tFidVec'], titleBOWTrain, titleBOWTest = functions.dataFromText(cleanedData['title'], maskTrainTest['maskTrain'], maskTrainTest['maskTest'], settings.tfidfParameters)

    # Include extracted text into training and testing
    variables.mlData['xTrain'] = functions.mergeDataFrames(xTrain, titleBOWTrain)
    variables.mlData['xTest'] = functions.mergeDataFrames(xTest, titleBOWTest)
    variables.mlData['yTrain'] = yTrain
    variables.mlData['yTest'] = yTest

    # Use the model and look at the scores
    variables.mlData['modelRF'], variables.mlData['probRF'], variables.mlData['apsRF'], variables.mlData['roc_aucRF'] = models.randomForestWMetrics(variables.mlData['xTrain'], variables.mlData['yTrain'], variables.mlData['xTest'], variables.mlData['yTest'])

    # store the cleaned data and features
    variables.mlData['cleanedData'] = cleanedData
    variables.mlData['features'] = features
    variables.mlData['y'] = y
