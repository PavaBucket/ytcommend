import functions

import pandas as pd


def clean():
    # Read Treated Data from CSV
    treatedData = pd.read_csv("./data/ytTreatedLinks.csv")
    treatedData = treatedData[treatedData['y'].notnull()]

    # Clean the data
    cleanedData = functions.clean(treatedData)

    # Create features
    features = functions.createFeatures(cleanedData)

    # Create the y series
    y = treatedData['y'].copy()

    # Train and test segmentation
    mlData = {}
    maskTrain = cleanedData['upload_date'] < '2020-12-08'
    maskTest = cleanedData['upload_date'] >= '2020-12-08'
    mlData['xTrain'], mlData['xTest'] = features[maskTrain], features[maskTest]
    mlData['yTrain'], mlData['yTest'] = y[maskTrain], y[maskTest]

    # Extract data from text
    mlData = functions.dataFromText(cleanedData, maskTrain, maskTest, mlData)

    return mlData
