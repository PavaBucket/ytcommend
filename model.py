import numpy as np
import pandas as pd
from matplotlib import pylab
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Read Treated Data from CSV
treatedData = pd.read_csv("./data/ytTreatedLinks.csv")
treatedData = treatedData[treatedData['y'].notnull()]

# Clean the data
cleanedData = pd.DataFrame(index=treatedData.index)
cleanedData['title'] = treatedData['title']
cleanedData['upload_date'] = pd.to_datetime(treatedData['upload_date'], format="%Y%m%d")
cleanedData['view_count'] = treatedData['view_count']

# Create the features dataset
features = pd.DataFrame(index=cleanedData.index)
features['time_since_up'] = (pd.to_datetime('2021-02-10') - cleanedData['upload_date']) / np.timedelta64(1, 'D')
features['view_count'] = cleanedData['view_count']
features['views_day'] = features['view_count'] / features['time_since_up']
features = features.drop(['time_since_up'], axis=1)

# Create the y series
y = treatedData['y'].copy()

# Train and test segmentation
maskTrain = cleanedData.index < 200
maskTest = cleanedData.index >= 200
xTrain, xTest = features[maskTrain], features[maskTest]
yTrain, yTest = y[maskTrain], y[maskTest]

# Active Learning: Extracting text to help labeling
titleTrain = cleanedData[maskTrain]['title']
titleTest = cleanedData[maskTest]['title']
titleVec = TfidfVectorizer(min_df=2)
titleBOWTrain = titleVec.fit_transform(titleTrain)
titleBOWTest = titleVec.transform(titleTest)

# Include extracted text into training and testing
xTrainWText = hstack([xTrain, titleBOWTrain])
xTestWText = hstack([xTest, titleBOWTest])

# Modelling: model 1
model1 = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight="balanced")
model1.fit(xTrainWText, yTrain)

prob1 = model1.predict_proba(xTestWText)[:, 1]

# Modelling: model 2
model2 = RandomForestClassifier(n_estimators=1000, random_state=0, class_weight="balanced", n_jobs=6)
model2.fit(xTrainWText, yTrain)

prob2 = model2.predict_proba(xTestWText)[:, 1]

# Metrics for testing the model
average_precision_score(yTest, prob1)
roc_auc_score(yTest, prob1)

fig, ax = pylab.subplots(1, 1, figsize=(20, 20))
plot_tree(model1, ax=ax, feature_names=xTrain.columns)
