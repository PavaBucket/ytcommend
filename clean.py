import functions

import pandas as pd
import settings


def clean():
    # Read Treated Data from CSV
    treatedData = pd.read_csv(settings.treatedLinksPath)
    treatedData = treatedData[treatedData['y'].notnull()]

    # Clean the data
    cleanedData = functions.clean(treatedData)

    # Create features
    features = functions.createFeatures(cleanedData)

    # Create the y series
    y = treatedData['y'].copy()

    # Train and test segmentation
    mlData = {}
    maskTrainTest = settings.getMaskTrainTest(cleanedData)
    mlData['xTrain'], mlData['xTest'] = features[maskTrainTest['maskTrain']], features[maskTrainTest['maskTest']]
    mlData['yTrain'], mlData['yTest'] = y[maskTrainTest['maskTrain']], y[maskTrainTest['maskTest']]

    # Extract data from text
    mlData = functions.dataFromText(cleanedData, maskTrainTest['maskTrain'], maskTrainTest['maskTest'], mlData, settings.tfidfParameters)

    return mlData
