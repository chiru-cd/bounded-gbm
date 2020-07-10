import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def set_reason(test, bounds):
    columns = list(bounds)

    for i in columns:
        low = bounds.at[0, i]
        high = bounds.at[1, i]
        test.loc[(test[i] < low), "bound_"+i] = "low_" + str(low-test[i])
        test.loc[(test[i] > high), "bound_"+i] = "high_" + str(test[i]-high)
        test.loc[(test[i] >= low) & (test[i] <= high), "bound_"+i] = 0
    return test

def get_reason_matrix(test, bounds):
    test = set_reason(test, bounds)
    columns = list(bounds)
    reason = pd.DataFrame()
    
    for i in columns:
        reason[i] = test["bound_"+i]
    return reason

def get_result(test, bounds):
    columns = list(bounds)
    reason = set_reason(test, bounds)
    test['unbound'] = 0
    
    for i in columns:
        test['unbound'] = np.where(test["bound_"+i]==0, test['unbound'], test['unbound']+1)
        test.pop("bound_"+i)
    return test

def evaluate(test, model, bounds):
    test0 = test.copy()
    y_test0 = test0.pop('class')
    X_y_test0 = xgb.DMatrix(data=test0)

    preds0 =  model.predict(X_y_test0)
    print (roc_auc_score(y_test0, preds0))

    test = get_result(test, bounds)
    test = test[test['unbound']==0]
    test.pop('unbound')

    y_test = test.pop('class')
    X_y_test = xgb.DMatrix(data=test)

    preds =  model.predict(X_y_test)
    print (roc_auc_score(y_test, preds))