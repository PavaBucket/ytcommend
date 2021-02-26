import pandas as pd
from scipy.sparse import hstack

import functions
import settings
import variables


def actLearning():

    # Unlabeled data #

    # Get the unlabeled data from csv
    unlabeledData = pd.read_csv(settings.treatedLinksPath)
    unlabeledData = unlabeledData[unlabeledData['y'].isnull()].dropna(how='all')

    # Clean the data
    cleanedData = functions.clean(unlabeledData)

    # Create features
    features = functions.createFeatures(cleanedData)

    # Use the data extractor from text
    textVector = variables.mlData['tFidVec'].transform(cleanedData['title'])

    # Include extracted text into features
    features = hstack([features, textVector])

    # Using the first model, predict the results for the unlabeled data and store on features
    probRFUnlabeled = variables.mlData['modelRF'].predict_proba(features)[:, 1]
    unlabeledData['p'] = probRFUnlabeled

    # Find the most difficult predictions, get some random examples and put in csv to label
    maskUnlabeled = settings.getActiveLearningFilters(unlabeledData)
    hardExamples = unlabeledData[maskUnlabeled]
    randomExamples = unlabeledData[~maskUnlabeled].sample(30)
    pd.concat([hardExamples, randomExamples]).to_csv(settings.actLearningExamplesPath)
