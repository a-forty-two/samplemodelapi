import pandas as pd
from sklearn.model_selection import train_test_split
from autosklearn.classification import AutoSklearnClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
def train_model(dataset_path):
  data = pd.read_csv(dataset_path)
  data = data.set_index('id')
  x = data.loc[:, ['radius_mean','area_mean']]
  y = data.loc[:, ['diagnosis']]
  logic = lambda val: 1 if val=='M' else 0
  y['diagnosis'] = [ logic(d) for d in y['diagnosis']]
  xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=42)
  model = AutoSklearnClassifier(time_left_for_this_task=120,
    per_run_time_limit=30)
  model.fit(xtrain, ytrain)
  dump(model, 'model.joblib') 

train_model('data.csv')
