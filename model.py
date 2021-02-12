import numpy as np
import pandas as pd
from matplotlib import pylab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
maskVal = cleanedData.index >= 200
xtrain, xtest = features[maskTrain], features[maskVal]
ytrain, ytest = y[maskTrain], y[maskVal]

# Modelling
model = DecisionTreeClassifier(random_state=0, max_depth=3, class_weight="balanced")
model.fit(xtrain, ytrain)

prob = model.predict_proba(xtest)[:, 1]

# Metrics for testing the model
average_precision_score(ytest, prob)
roc_auc_score(ytest, prob)

fig, ax = pylab.subplots(1, 1, figsize=(20, 20))
plot_tree(model, ax=ax, feature_names=xtrain.columns)
