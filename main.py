# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import operator
from sklearn.metrics import roc_auc_score

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

# change = np.array([-300,-290,-280,-270,-260,-250,-240,-230,-220,-210,-200,-190,-180,-170,-160,-150,-140,-130,-120,-110,-100,-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300])
# change = np.array([-90,-80,-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300])
# ser = pd.Series(change)
# df['percent_change'] = ser

for i in featdict:
    a = [0] * 32
    b = [0] * 32
    flaglt = 1
    flaggt = 1
    aprev = 2
    bprev = 2
    for j in range (0, 31):
        lt = train.copy()
        gt = train.copy()

        lt[i] = lt[i].apply(lambda x: x - (j/100)*x)
        gt[i] = gt[i].apply(lambda x: x + (j/100)*x)

        ylt = lt.pop('class')
        Xlt = lt
        ygt = gt.pop('class')
        Xgt = gt

        # creating Dmatrix of the modified data
        X_test_lt = xgb.DMatrix(data=Xlt)
        X_test_gt = xgb.DMatrix(data=Xgt)

        # performing prediction on new test dataset
        predslt =  model.predict(X_test_lt)
        predsgt =  model.predict(X_test_gt)

        a[j] = np.mean(predslt)
        b[j] = np.mean(predsgt)
        
        if j!=0:
            if (abs(a[j]-aprev) < 0.01) and flaglt==1:
                a[31] = (j)
                flaglt = 0

            if (abs(b[j]-bprev) < 0.01) and flaggt==1:
                b[31] = (j)
                flaggt = 0
        
        aprev=a[j]
        bprev=b[j]

    if flaglt==1:
        a[31]=30

    if flaggt==1:
        b[31]=30

    serlt = pd.Series(a)
    sergt = pd.Series(b)
    colnamelt = i+ "_dec"
    colnamegt = i+ "_inc"
    df[colnamelt] = serlt
    df[colnamegt] = sergt

# df.iloc[0:9].plot()
# df.plot(x='percent_change')
# plt.show()
print (df)

test0 = test.copy()
y_test = test0.pop('class')
X_test = test0
X_y_test = xgb.DMatrix(data=X_test)

preds =  model.predict(X_y_test)
print (test0.shape)
print (roc_auc_score(y_test, preds))

for i in featdict:
    mint = df.at[31,i+ "_dec"]
    maxt = df.at[31,i+ "_inc"]
    test.drop(test[test[i] < ((1+mint/2)*train[i].min())].index, inplace = True) 
    test.drop(test[test[i] > ((1-maxt/2)*train[i].max())].index, inplace = True) 

print (test.shape)
y_test = test.pop('class')
X_test = test
X_y_test = xgb.DMatrix(data=X_test)

preds =  model.predict(X_y_test)
print (roc_auc_score(y_test, preds))