# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

l1 = pd.read_csv("loan_2018Q4.csv")
l2 = pd.read_csv("loan_2019Q1.csv")
l3 = pd.read_csv("loan_2019Q2.csv")
l4 = pd.read_csv("loan_2019Q3.csv")
# loan = l1.append([l2,l3], ignore_index=True)
train = l1
test = l4
y_train = train.pop('class')
X_train = train
y_test = test.pop('class')
X_test = test

# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42, stratify=y)

print(X_train.shape)

X_y_train = xgb.DMatrix(data=X_train, label= y_train)
X_y_test = xgb.DMatrix(data=X_test)

params = {
          'base_score': np.mean(y_train),
          'eta': 0.1,
          'max_depth': 5,
          'gamma' :3,
          'objective'   :'binary:logistic',
          'eval_metric' :'auc'
         }

model = xgb.train(params=params, dtrain=X_y_train, num_boost_round=500)

preds =  model.predict(X_y_test)

print (roc_auc_score(y_test, preds))
pickle.dump(model, open("test.dat", "wb"))
