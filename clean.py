import functions

import pandas as pd
import settings
import variables


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
    maskTrainTest = settings.getMaskTrainTest(cleanedData)
    variables.mlData['xTrain'], variables.mlData['xTest'] = features[maskTrainTest['maskTrain']], features[maskTrainTest['maskTest']]
    variables.mlData['yTrain'], variables.mlData['yTest'] = y[maskTrainTest['maskTrain']], y[maskTrainTest['maskTest']]

    # Extract data from text
    variables.mlData = functions.dataFromText(cleanedData, maskTrainTest['maskTrain'], maskTrainTest['maskTest'], variables.mlData, settings.tfidfParameters)
