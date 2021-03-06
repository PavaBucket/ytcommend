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
    xTrain, xTest = features[maskTrainTest['maskTrain']], features[maskTrainTest['maskTest']]
    yTrain, yTest = y[maskTrainTest['maskTrain']], y[maskTrainTest['maskTest']]

    # Extract data from text
    variables.mlData['tFidVec'], titleBOWTrain, titleBOWTest = functions.dataFromText(cleanedData['title'], maskTrainTest['maskTrain'], maskTrainTest['maskTest'], settings.tfidfParameters)

    # Include extracted text into training and testing
    variables.mlData['xTrain'] = functions.mergeDataFrames(xTrain, titleBOWTrain)
    variables.mlData['xTest'] = functions.mergeDataFrames(xTest, titleBOWTest)
    variables.mlData['yTrain'] = yTrain
    variables.mlData['yTest'] = yTest
