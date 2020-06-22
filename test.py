# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_y_train = xgb.DMatrix(data=train[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']], label= train['Survived'])
X_test    = xgb.DMatrix(data=test[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])

params = {
          'base_score': np.mean(train['Survived']),
          'eta':  0.1,
          'max_depth': 3,
          'gamma' :3,
          'objective'   :'reg:squarederror',
          'eval_metric' :'mae'
         }
model = xgb.train(params=params, dtrain=X_y_train, num_boost_round=3)
pickle.dump(model, open("test.dat", "wb"))