# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from split import split

print ("Make sure the model & data files are in the working directory")
model_in = input ("Enter model file (.dat) ")
data_in = input ("Enter training data file (.csv) ")

split(open('/home/chirag/bounded-gbm/data/' + data_in, 'r'))

data = pd.read_csv('output_1.csv')
model = pickle.load(open(model_in, "rb"))
featdict = model.get_fscore()
for i in featdict:
    print(train[[i]].describe())
    print(model.get_split_value_histogram(i))