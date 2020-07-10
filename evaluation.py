import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, accuracy_score, precision_score, recall_score, confusion_matrix
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
    auc_orig = (roc_auc_score(y_test0, preds0))
    mse_orig = (mean_squared_error(y_test0, preds0))
    log_orig = (log_loss(y_test0, preds0))

    testu = get_result(test, bounds)
    test = testu[testu['unbound']==0]
    testu = testu[testu['unbound']>0]
    
    test.pop('unbound')
    testu.pop('unbound')
    
    y_testu = testu.pop('class')
    X_y_testu = xgb.DMatrix(data=testu)

    predsu =  model.predict(X_y_testu)

    tn, fp, fn, tp = confusion_matrix(y_testu, predsu.round()).ravel()
    print ("In unbounded, tn =",tn,"fp =",fp,"fn =",fn,"tp =",tp)
    acc_u = accuracy_score(y_testu, predsu.round())
    pre_u = precision_score(y_testu, predsu.round())
    rec_u = recall_score(y_testu, predsu.round())

    # print("In", len(testu), "unbound records,", acc_u,"were correctly classified")
    print("Accuracy of unbounded:", acc_u)
    print("Precision of unbounded:", pre_u)
    print("Recall of unbounded:", rec_u)
    
    y_test = test.pop('class')
    X_y_test = xgb.DMatrix(data=test)

    preds =  model.predict(X_y_test)
    auc_new = (roc_auc_score(y_test, preds))
    mse_new = (mean_squared_error(y_test, preds))
    log_new = (log_loss(y_test, preds))
    
    print ("AUC score:", ((auc_new-auc_orig)/auc_orig)*100,"%")
    print ("Mean Squared Error:", ((mse_new-mse_orig)/mse_orig)*100,"%")
    print ("Neg Log Loss:", ((log_new-log_orig)/log_orig)*100,"%")