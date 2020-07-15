# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

data_in = input ("Enter processed training data file")
path = os.getcwd()
loan = pd.read_csv(os.path.abspath(os.path.join(path, '..'))+'/'+data_in)

data_in2 = input ("Enter processed testing data file")
path = os.getcwd()
loan_test = pd.read_csv(os.path.abspath(os.path.join(path, '..'))+'/'+data_in2)

# l1 = pd.read_csv("loan_2019Q1.csv")
# l2 = pd.read_csv("loan_2019Q2.csv")
# l3 = pd.read_csv("loan_2019Q3.csv")
# l4 = pd.read_csv("loan_2018Q4.csv")
# loan = l1.append([l2,l3,l4], ignore_index=True)
# loan.to_csv("dev.csv", index=False)
train = loan
test = loan_test
y_train = train.pop('class')
X_train = train
y_test = test.pop('class')
X_test = test

# X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42, stratify=y)

X_y_train = xgb.DMatrix(data=X_train, label= y_train)
X_y_test = xgb.DMatrix(data=X_test)

params = {
          'base_score': np.mean(y_train),
          'eta': 0.1,
          'max_depth': 5,
          'gamma' :3,
          'objective' :'binary:logistic',
         }

model = xgb.train(params=params, dtrain=X_y_train, num_boost_round=500)

preds =  model.predict(X_y_test)

print (roc_auc_score(y_test, preds))
pickle.dump(model, open("../model.dat", "wb"))