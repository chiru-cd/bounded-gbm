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
#model_in = input ("Enter model file (.dat) ")
model_in = "test.dat"
# input training data
#data_in = input ("Enter training data file (.csv) ")
data_in = "test.csv"
# if input file is large
#subsplits = input ("Create subsplits (y/n) ")
subsplits = "n"
# input feature count
#feat_count = int( input ("Enter no. of features ") )
feat_count = 1

# splitting large csv into smaller output_i.csv keeping header intact
# loading data
if subsplits == 'y':
    split(open('data/' + data_in, 'r'))
    train = pd.read_csv('ouput_1.csv', low_memory=False)
else:
    # train = pd.read_csv('data/' + data_in, low_memory=False)
    train = pd.read_csv(data_in, low_memory=False)

# loading model
model = pickle.load(open(model_in, "rb"))

# getting 'n' most important features
def subdict (count):
    d = model.get_fscore()
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:count])
    return sorted_d

featdict = subdict(feat_count)

for i in featdict:
    df = pd.DataFrame(columns=['Fractional change', 'RMSE'])
    for j in range (10):
        lt = train.copy()
        gt = train.copy()
        lt[i] = lt[i].apply(lambda x: x - (j/10)*x)
        gt[i] = gt[i].apply(lambda x: x + (j/10)*x)
        train_mod = lt.append(gt)

        # creating Dmatrix of the modified data
        X_test = xgb.DMatrix(data=train_mod[['Age', 'Fare', 'SibSp', 'Parch']])

        # performing prediction on new test dataset
        preds =  model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(train_mod['pred'], preds))

        df.loc[j] = [j/10, rmse]

print (df)