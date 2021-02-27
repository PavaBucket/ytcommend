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

# Data files paths
rawLinksPath = "./data/ytRawLinks.json"
treatedLinksPath = "./data/ytTreatedLinks.csv"
actLearningExamplesPath = "./data/ytActiveLearningExamples.csv"

# Models paths
RandomForestPath = "./models/RandomForest.pkl.z"
lightGBMPath = "./models/LightGBM.pkl.z"
logisticRegressionPath = "./models/LogisticRegression.pkl.z"
VectorizerPath = "./models/Vectorizer.pkl.z"


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


# Optimization

# LGBM: Space boundaries for optimization

LGBMSpace = [
    (1e-3, 1e-1, 'log-uniform'),  # learning rate
    (1, 10),  # max_depth
    (1, 20),  # min_child_samples
    (0.05, 1.),  # subsample
    (0.05, 1.),  # colsample_bytree
    (100, 1000),  # n_estimators
    (1, 5),  # min_df
    (1, 5),  # ngram_range
]
