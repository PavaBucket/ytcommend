# Youtube-dl options
ydl_opts = {
    'quiet': True,
    'skip_download': True,
    'forceid': True,
    'forcetitle': True,
    'forceurl': True,
    'forcejson': True,
    'ignoreerrors': True,
    'download': False,
}

# Query filters for collecting data
baseQuery = "ytsearchdate100"
queryFilters = ["machine+learning", "data+science", "kaggle"]

# File paths
rawLinksPath = "./data/ytRawLinks.json"
treatedLinksPath = "./data/ytTreatedLinks.csv"
actLearningExamplesPath = "./data/ytActiveLearningExamples.csv"


# Train test split settings
def getMaskTrainTest(cleanedData):
    maskTrain = cleanedData['upload_date'] < '2020-12-08'
    maskTest = cleanedData['upload_date'] >= '2020-12-08'
    maskTrainTest = {'maskTrain': maskTrain, 'maskTest': maskTest}

    return maskTrainTest


# Parameters for the data extractor from text
tfidfParameters = {'min_df': 2, 'ngram_range': (1, 2)}


# Active learning query filters
def getActiveLearningFilters(unlabeledData):
    maskUnlabeled = (unlabeledData['p'] <= 0.58) & (unlabeledData['p'] >= 0.42)
    return maskUnlabeled


# Optimization #

# LGBM

# lgbmParameters = {'learning_rate': , 'max_depth': , 'min_child_samples': , 'subsample': , 'colsample_bytree': , 'n_estimators': , }
