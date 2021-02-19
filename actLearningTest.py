import pandas as pd
import functions
import models


def actLearningTest(mlData):

    # Read Treated Data from CSV
    treatedData = pd.read_csv("./data/ytTreatedLinks.csv")
    treatedData = treatedData[treatedData['y'].notnull()]

    # Get the new active learning labeled data
    actLearnData = pd.read_csv("./data/ytActiveLearningExamples.csv")
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

    # Test 1: increase test dataset
    maskTrain = (cleanedData['upload_date'] < '2020-12-08') & (cleanedData['new'] == 0)
    maskTest = cleanedData['upload_date'] >= '2020-12-08'
    mlData['xTrain'], mlData['xTest'] = features[maskTrain], features[maskTest]
    mlData['yTrain'], mlData['yTest'] = y[maskTrain], y[maskTest]

    # Extract data from text
    mlData = functions.dataFromText(cleanedData, maskTrain, maskTest, mlData)

    # Use the model and look at the scores
    mlData = models.randomForestWMetrics(mlData)

    # Test 2: increase train dataset
    maskTrain = (cleanedData['upload_date'] < '2020-12-08')
    maskTest = (cleanedData['upload_date'] >= '2020-12-08') & (cleanedData['new'] == 0)
    mlData['xTrain'], mlData['xTest'] = features[maskTrain], features[maskTest]
    mlData['yTrain'], mlData['yTest'] = y[maskTrain], y[maskTest]

    # Extract data from text
    mlData = functions.dataFromText(cleanedData, maskTrain, maskTest, mlData)

    # Use the model and look at the scores
    mlData = models.randomForestWMetrics(mlData)

    # Test 3: increase both datasets
    maskTrain = cleanedData['upload_date'] < '2020-12-08'
    maskTest = cleanedData['upload_date'] >= '2020-12-08'
    mlData['xTrain'], mlData['xTest'] = features[maskTrain], features[maskTest]
    mlData['yTrain'], mlData['yTest'] = y[maskTrain], y[maskTest]

    # Extract data from text
    mlData = functions.dataFromText(cleanedData, maskTrain, maskTest, mlData)

    # Use the model and look at the scores
    mlData = models.randomForestWMetrics(mlData)

    return mlData
