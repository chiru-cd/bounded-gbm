# git rm -r --cached data

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
# model_in = input ("Enter model file (.dat) ")
model_in = "test.dat"
# input training data
# data_in = input ("Enter training data file (.csv) ")
data_in = "loan_2018Q4.csv"
# input testing data
# data_in2 = input ("Enter testing data file (.csv) ")
data_in2 = ["loan_2016Q1", "loan_2016Q2","loan_2016Q3", "loan_2016Q4","loan_2017Q1", "loan_2017Q2","loan_2017Q3", "loan_2017Q4","loan_2018Q1", "loan_2018Q2"]
# input tolerance value
toler = int(input ("Enter tolerance value (%) "))
# input feature count
feat_count = int( input ("Enter no. of features ") )

# loading data
train = pd.read_csv(data_in)
# test = pd.read_csv(data_in2)

# loading model
model = pickle.load(open(model_in, "rb"))

# excel output path
path = r"/home/chirag/bounded-gbm/output.xlsx"

# getting 'n' most important features
def subdict (count):
    d = model.get_fscore()
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:count])
    return sorted_d

featdict = subdict(feat_count)

df = pd.DataFrame(index=['min_split_value', 'max_split_value', 'weighted_mean_splits', 'min_feature_value', 'max_feature_value', 'mean_feature_value', 'test_lt_minsplit', 'test_gt_maxsplit'])
df_percent_min = pd.DataFrame(index=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])
df_percent_max = pd.DataFrame(index=['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','90-100'])

for i in featdict:
    a = [0] * 8
    feat_df = model.get_split_value_histogram(i)
    a[0] = feat_df['SplitValue'].min()
    a[1] = feat_df['SplitValue'].max()
    a[2] = np.average(feat_df['SplitValue'], weights=feat_df['Count'])
    a[3] = train[i].min()
    a[4] = train[i].max()
    a[5] = train[i].mean()
    
    dfmin = train[train[i] < a[0]]
    df_percent_min[i] = dfmin[i].value_counts(bins=10, sort=False).values
    
    dfmax = train[train[i] > a[1]]
    df_percent_max[i] = dfmax[i].value_counts(bins=10, sort=False).values
    
    a[6] = dfmin.shape[0]
    a[7] = dfmax.shape[0]
    serlt = pd.Series(a)
    df[i] = serlt.values

auc_percent = pd.DataFrame(index=['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%','unbounded'])
num_rec_percent = pd.DataFrame(index=['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%','unbounded'])
# testing original test dataset

for x in data_in2:
    test = pd.read_csv(x+".csv")
    b = [0] * 12
    c = [0] * 12
    for j in range(12):
        test1 = test.copy()
        if j!=11:
            for i in featdict:
                mint = df.at['min_split_value', i]
                maxt = df.at['max_split_value', i]
                test1.drop(test1[test1[i] < (1-(j/10))*mint].index, inplace = True) 
                test1.drop(test1[test1[i] > (1+(j/10))*maxt].index, inplace = True) 

        c[j] = len(test) - len(test1)
        y_test = test1.pop('class')
        X_test = test1
        X_y_test = xgb.DMatrix(data=X_test)

        preds =  model.predict(X_y_test)
        b[j] = (roc_auc_score(y_test, preds))

    serb = pd.Series(b)
    auc_percent[x] = serb.values

    serc = pd.Series(c)
    num_rec_percent[x] = serc.values


writer = pd.ExcelWriter(path)
df.to_excel(writer, sheet_name = 'Stats')
df_percent_min.to_excel(writer, sheet_name = 'min_stats')
df_percent_max.to_excel(writer, sheet_name = 'max_stats')
auc_percent.to_excel(writer, sheet_name = 'auc_scores')
num_rec_percent.to_excel(writer, sheet_name = 'num_recs_dropped')

writer.save()
writer.close()