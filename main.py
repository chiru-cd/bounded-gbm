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
# if input file is large
subsplits = input ("Create subsplits (y/n) ")
# input feature count
feat_count = int( input ("Enter no. of features ") )

# splitting large csv into smaller output_i.csv keeping header intact
# loading data
if subsplits == 'y':
    split(open(data_in, 'r'))
    train = pd.read_csv('output_1.csv', low_memory=False)
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

# iterating over features by RMSE value
# for i in featdict:
#     flag = 1
#     for j in range (10):
#         lt = train.copy()
#         gt = train.copy()
#         lt[i] = lt[i].apply(lambda x: x - (j/10)*x)
#         gt[i] = gt[i].apply(lambda x: x + (j/10)*x)
#         train_mod = lt.append(gt)

#         y = train_mod.pop('class')
#         X = train_mod
#         # creating Dmatrix of the modified data
#         X_test = xgb.DMatrix(data=train_mod)

#         # performing prediction on new test dataset
#         preds =  model.predict(X_test)

#         rmse = np.sqrt(mean_squared_error(train_mod[label], preds))

#         if j==0:
#             rmse_orig = rmse
        
#         if rmse < (0.99)*rmse_orig or rmse > (1.01)*rmse_orig:
#             print("Threshold for "+i+" is +/-",(j/10),"i.e.", (1-j/10)*train[i].min(),"-",(1+j/10)*train[i].max())
#             flag = 0
#             break
    
#     if flag == 1:
#         print("Model is insensitive to "+i)

df = pd.DataFrame()

for i in featdict:
    a = [0] * 11
    b = [0] * 11
    flaglt = 1
    flaggt = 1
    for j in range (10):
        lt = train.copy()
        gt = train.copy()
        lt[i] = lt[i].apply(lambda x: x - (j/10)*x)
        gt[i] = gt[i].apply(lambda x: x + (j/10)*x)

        ylt = lt.pop('class')
        ygt = gt.pop('class')
        Xlt = lt
        Xgt = gt

        # creating Dmatrix of the modified data
        X_test_lt = xgb.DMatrix(data=Xlt)
        X_test_gt = xgb.DMatrix(data=Xgt)

        # performing prediction on new test dataset
        predslt =  model.predict(X_test_lt)
        predsgt =  model.predict(X_test_gt)

        a[j] = np.mean(predslt)
        b[j] = np.mean(predsgt)

        if j==0:
            pred_orig = a[j]
        
        if (a[j] < (0.9)*pred_orig or a[j] > (1.1)*pred_orig) and flaglt==1:
            a[10] = (j*100)
            flaglt = 0

        if (b[j] < (0.9)*pred_orig or b[j] > (1.1)*pred_orig) and flaggt==1:
            b[10] = (j*100)
            flaggt = 0
    
    if flaglt==1:
        a[10]=np.NaN

    if flaggt==1:
        b[10]=np.NaN

    serlt = pd.Series(a)
    sergt = pd.Series(b)
    colnamelt = i+ " dec"
    colnamegt = i+ " inc"
    df[colnamelt] = serlt
    df[colnamegt] = sergt

df.iloc[0:9].plot()
plt.show()
print (df)