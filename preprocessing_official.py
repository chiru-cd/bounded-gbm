import pandas as pd
import numpy as np

data = pd.read_csv("data/Loan_2018Q4.csv", low_memory=False)

data['int_rate'] = [float(t.rstrip('%')) for t in data['int_rate']]

def convt(t):
    try:
        return float(t.rstrip('%'))
    except:
        return t

data['revol_util'] = [convt(k) for k in data['revol_util']]

data_missing = data.isna()
data_missing_count = data_missing.sum()

data_missing_percentage = (data_missing_count / len(data)).round(4) * 100
data_missing_sorted = data_missing_percentage.sort_values(ascending=False)
(data_missing_sorted.head(60))

#del all col which has more than 70% missing
temp = [i for i in data.count()<len(data)*0.70]
data.drop(data.columns[temp],axis=1,inplace=True)

data.loc[(data['loan_status'] == 'Current') | (data['loan_status'] == 'Fully Paid'), 'class'] = 1
data.loc[(data['loan_status'] != 'Current') & (data['loan_status'] != 'Fully Paid'), 'class'] = 0

col_to_drop = ["id"]
data = data.select_dtypes(exclude=['object'])
data.drop(col_to_drop, axis = 1, inplace=True)

print (data['class'].value_counts())
data.to_csv("loan_2018Q4.csv", index=False)