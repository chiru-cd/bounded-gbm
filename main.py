# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import operator
from split import split

# Make sure the model & data files are in the working directory
# input model
model_in = input ("Enter model file (.dat) ")
# input training data
data_in = input ("Enter training data file (.csv) ")
# if input file is large
subsplits = input ("Create subsplits (y/n) ")
# input feature count
feat_count = int( input ("Enter no. of features ") )

# splitting large csv into smaller output_i.csv keeping header intact
if subsplits == 'y':
    split(open('data/' + data_in, 'r'))

# load data
data = pd.read_csv('data/' + data_in, low_memory=False)

# load model
model = pickle.load(open(model_in, "rb"))

def subdict (count):
    d = model.get_fscore()
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:count])
    return sorted_d

featdict = subdict(feat_count)
for i in featdict:
    print(data[[i]].describe())
    print(model.get_split_value_histogram(i))