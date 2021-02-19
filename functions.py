import pandas as pd
import numpy as np

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


# Treat the data
def treat(data):
    data = data.drop_duplicates(subset=['webpage_url'])
    return data


# Clean the data
def clean(data):
    cleanedData = pd.DataFrame(index=data.index)
    cleanedData['title'] = data['title']
    cleanedData['upload_date'] = pd.to_datetime(data['upload_date'], format="%Y%m%d")
    cleanedData['view_count'] = data['view_count']
    return cleanedData


# Make the features dataset
def createFeatures(cleanedData):
    features = pd.DataFrame(index=cleanedData.index)
    features['time_since_up'] = (pd.to_datetime('2021-02-10') - cleanedData['upload_date']) / np.timedelta64(1, 'D')
    features['view_count'] = cleanedData['view_count']
    features['views_day'] = features['view_count'] / features['time_since_up']
    features = features.drop(['time_since_up'], axis=1)
    return features


# Extract data from text
def dataFromText(cleanedData, maskTrain, maskTest, mlData, tfidfParameters):
    titleTrain = cleanedData[maskTrain]['title']
    titleTest = cleanedData[maskTest]['title']
    titleVec = TfidfVectorizer(tfidfParameters)
    titleBOWTrain = titleVec.fit_transform(titleTrain)
    titleBOWTest = titleVec.transform(titleTest)

    # Include extracted text into training and testing
    xTrainWText = hstack([mlData['xTrain'], titleBOWTrain])
    xTestWText = hstack([mlData['xTest'], titleBOWTest])

    mlData['tFidVec'] = titleVec
    mlData['xTrain'] = xTrainWText
    mlData['xTest'] = xTestWText

    return mlData
