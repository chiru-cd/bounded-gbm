# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

# model = xgb.Booster({'nthread': 4})  # init model
# model.load_model('0003.model')  # load data
train = pd.read_csv("data/train.csv")
model = pickle.load(open("test.dat", "rb"))
featdict = model.get_fscore()
for i in featdict:
    print(train[[i]].describe())
    print(model.get_split_value_histogram(i))
