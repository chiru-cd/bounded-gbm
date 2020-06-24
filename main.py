# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import operator
from split import split
from sklearn.metrics import mean_squared_error

# Make sure the model & data files are in the working directory
# input model
model_in = input ("Enter model file (.dat) ")
# input training data
data_in = input ("Enter training data file (.csv) ")
# input target feature
label = input ("Enter target label ")
# if input file is large
subsplits = input ("Create subsplits (y/n) ")
# input feature count
feat_count = int( input ("Enter no. of features ") )

# splitting large csv into smaller output_i.csv keeping header intact
# loading data
if subsplits == 'y':
    split(open(data_in, 'r'))
    train = pd.read_csv('ouput_1.csv', low_memory=False)
else:
    train = pd.read_csv(data_in, low_memory=False)

# loading model
model = pickle.load(open(model_in, "rb"))

# getting 'n' most important features
d = model.get_fscore()

def subdict (count):
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:count])
    return sorted_d

featdict = subdict(feat_count)

# iterating over features
for i in featdict:
    flag = 1
    for j in range (10):
        lt = train.copy()
        gt = train.copy()
        lt[i] = lt[i].apply(lambda x: x - (j/10)*x)
        gt[i] = gt[i].apply(lambda x: x + (j/10)*x)
        train_mod = lt.append(gt)

        # creating Dmatrix of the modified data
        X_test = xgb.DMatrix(data=train_mod[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']])

        # performing prediction on new test dataset
        preds =  model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(train_mod[label], preds))

        if j==0:
            rmse_orig = rmse
        
        if rmse < (0.99)*rmse_orig or rmse > (1.01)*rmse_orig:
            print("Threshold for "+i+" is +/-",(j/10),"i.e.", (1-j/10)*train[i].min(),"-",(1+j/10)*train[i].max())
            flag = 0
            break
    
    if flag == 1:
        print("Model is insensitive to "+i)