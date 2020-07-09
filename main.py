# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import operator
from sklearn.metrics import roc_auc_score

from initialisation import createBase
from simulation import simulate
from experimentation import get_bounds
import evaluation as ev

# Make sure the model & data files are in the working directory
# input model
model_in = input ("Enter model file (.dat) ")
# input training data
data_in = input ("Enter training data file (.csv) ")
# input testing data
data_in2 = input ("Enter testing data file (.csv) ")
# input feature count
feat_count = int( input ("Enter no. of features ") )

# loading data
train = pd.read_csv(data_in)
test = pd.read_csv(data_in2)

# loading model
model = pickle.load(open(model_in, "rb"))

# getting 'n' most important features
def subdict (count):
    d = model.get_fscore()
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:count])
    return sorted_d

featdict = subdict(feat_count)

base = createBase(train)

bounds = pd.DataFrame()

for i in featdict:
    based = simulate(base, train, i)
    bounds[i] = get_bounds(based, model, i)
    
ev.evaluate(test, model, bounds)