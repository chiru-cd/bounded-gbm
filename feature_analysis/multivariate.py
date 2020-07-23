import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def monotone(train, model, poslist, neglist, delta):
    df = pd.DataFrame()
    a = [0] * 30
    
    varLists = [] * (len(poslist)+len(neglist))

    train_copy = train.sample(n = 25000)
    for j in range (0, 30):
        
        for i in range(len(poslist)):
            min_val = train[poslist[i]].min()
            max_val = train[poslist[i]].max()
            diff = (max_val - min_val)
            new_val = min_val + ((j/10)-1)*diff
            if 'fico' in poslist[i]:
                if new_val < 0:
                    new_val = 0
                if new_val > 900:
                    new_val = 900
            train_copy[poslist[i]] = new_val
            varLists[i].append(new_val)

        for i in range(len(neglist)):
            min_val = train[neglist[i]].min()
            max_val = train[neglist[i]].max()
            diff = (max_val - min_val)
            new_val = max_val - ((j/10)-1)*diff
            if 'fico' in i:
                if new_val < 0:
                    new_val = 0
                if new_val > 900:
                    new_val = 900
            train_copy[neglist[i]] = new_val
            varLists[len(poslist)+i].append(new_val)

        ylt = train_copy.pop('class')
        # creating Dmatrix of the modified data
        X_test_lt = xgb.DMatrix(data=train_copy)
        # performing prediction on new test dataset
        predslt =  model.predict(X_test_lt)

        a[j] = np.mean(predslt)

    for i in range(len(poslist)):
        df[poslist[i]] = varLists[i]
    for i in range(len(neglist)):
        df[neglist[i]] = varLists[len(poslist)+i]

    df["mean_preds"] = pd.Series(a)
    df['preds_diff'] = df['mean_preds'].diff()
    df['preds_diff'] = df['preds_diff'].apply(lambda x: -x if x<0 else x)
    indices = df.index[df['preds_diff'] > delta].tolist()
    lowi = indices[0]-1
    highi = indices[-1]

    bounds = pd.DataFrame()
    for i in poslist:
        low = df.at[lowi, i]
        high = df.at[highi, i]
        bounds[i] = pd.Series([low, high])
    for i in neglist:
        low = df.at[lowi, i]
        high = df.at[highi, i]
        bounds[i] = pd.Series([low, high])

    print (df)
    # df.plot()
    # plt.show()