import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def get_reason(test, bounds):
    columns = list(bounds)
    reason = pd.DataFrame()
    for i in columns:
        low = bounds[i][0]
        high = bounds[i][1]
        reason.loc[(test[i] < low), i] = "low_" + str(low-test[i])
        reason.loc[(test[i] > high), i] = "high_" + str(test[i]-high)
        reason.loc[(test[i] >= low) & (test[i] <= high), i] = 0
    
    return reason

def get_result(test, bounds):
    columns = list(bounds)
    reason = get_reason(test, bounds)
    test['unbound'] = 0
    for i in columns:
        test['unbound'] = np.where(isinstance(reason[i], str), test['unbound']+1, test['unbound'])
    
    return test

def evaluate(test, model, bounds):
    test0 = test.copy()
    y_test0 = test0.pop('class')
    X_y_test0 = xgb.DMatrix(data=test0)

    preds0 =  model.predict(X_y_test0)
    print (roc_auc_score(y_test0, preds0))

    test = get_result(test, bounds)
    test = test[test['unbound']!=0]
    test.pop('unbound')

    y_test = test.pop('class')
    X_y_test = xgb.DMatrix(data=test)

    preds =  model.predict(X_y_test)
    print (roc_auc_score(y_test, preds))