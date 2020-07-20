import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss, accuracy_score, precision_score, recall_score, confusion_matrix
from feature_analysis.initialisation import createBase

def eval_bound(train, model, bounds):
    columns = list(bounds)
    result = pd.DataFrame()
    for i in columns:
        base = createBase(train)
        base = base.sample(n=5000)
        low = bounds.at[0, i]
        high = bounds.at[1, i]
        min_val = train[i].min()
        max_val = train[i].max()
        diff = (max_val - min_val)/10
        a = [0] * 20
        b = [0] * 20

        for j in range(20):
            low_range = pd.Series(np.random.randint(low-(j+1)*diff, low-1-(j)*diff, size=5000))
            base[i] = low_range.values
            X_test = xgb.DMatrix(data = base)
            a[j] = np.mean(model.predict(X_test))

            high_range = pd.Series(np.random.randint(high+1+(j)*diff, high+(j+1)*diff, size=5000))
            base[i] = high_range.values
            X_test = xgb.DMatrix(data = base)        
            b[j] = np.mean(model.predict(X_test))

        result[i+"_low"] = pd.Series(a)
        result[i+"_high"] = pd.Series(b)
    
    return result

def set_reason(test, bounds):
    columns = list(bounds)

    for i in columns:
        low = bounds.at[0, i]
        high = bounds.at[1, i]
        test.loc[(test[i] < low) | (test[i] > high), "bound_"+i] = 1
        test.loc[(test[i] >= low) & (test[i] <= high), "bound_"+i] = 0
    return test

def get_reason_matrix(test, bounds):
    columns = list(bounds)
    reason = pd.DataFrame()

    for i in columns:
        low = bounds.at[0, i]
        high = bounds.at[1, i]
        mid = (low+high)/2

        test.loc[(test[i] < mid), "boundary_"+i] = "low"
        test.loc[(test[i] < mid), "margin_"+i] = test[i]-low
        test.loc[(test[i] >= mid), "boundary_"+i] = "high"
        test.loc[(test[i] >= mid), "margin_"+i] = high-test[i]
        
        reason["boundary_"+i] = test["boundary_"+i]
        reason["margin_"+i] = test["margin_"+i]
        test.pop("boundary_"+i)
        test.pop("margin_"+i)
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
    # original test dataset
    test0 = test.copy()
    y_test0 = test0.pop('class')
    X_y_test0 = xgb.DMatrix(data=test0)
    preds0 =  model.predict(X_y_test0)

    tn0, fp0, fn0, tp0 = confusion_matrix(y_test0, preds0.round()).ravel()

    auc_orig = (roc_auc_score(y_test0, preds0))
    mse_orig = (mean_squared_error(y_test0, preds0))
    log_orig = (log_loss(y_test0, preds0))
    pre_orig = precision_score(y_test0, preds0.round())
    fp_orig = fp0/len(test0)

    test = get_result(test, bounds)
    testu = test[test['unbound']>0]
    testb = test[test['unbound']==0]
    
    # unbounded records
    if len(testu) > 0:
        testu.pop('unbound')
        y_testu = testu.pop('class')
        X_y_testu = xgb.DMatrix(data=testu)

        predsu =  model.predict(X_y_testu)

        tnu, fpu, fnu, tpu = confusion_matrix(y_testu, predsu.round()).ravel()
        # print ("In unbounded, tn =",tnu,"fp =",fpu,"fn =",fnu,"tp =",tpu)
        acc_u = accuracy_score(y_testu, predsu.round())
        pre_u = precision_score(y_testu, predsu.round())
        rec_u = recall_score(y_testu, predsu.round())

    # print("In", len(testu), "unbound records,", acc_u,"were correctly classified")
    # print("Accuracy of unbounded:", acc_u)
    # print("Precision of unbounded:", pre_u)
    # print("Recall of unbounded:", rec_u)
    
    # bounded records
    testb.pop('unbound')
    y_testb = testb.pop('class')
    X_y_testb = xgb.DMatrix(data=testb)

    predsb =  model.predict(X_y_testb)

    tnb, fpb, fnb, tpb = confusion_matrix(y_testb, predsb.round()).ravel()

    auc_new = (roc_auc_score(y_testb, predsb))
    mse_new = (mean_squared_error(y_testb, predsb))
    log_new = (log_loss(y_testb, predsb))
    pre_new = precision_score(y_testb, predsb.round())
    fp_new = fpb/len(test0)

    print ("AUC score:", ((auc_new-auc_orig)/auc_orig)*100,"%")
    # print ("Mean Squared Error:", ((mse_new-mse_orig)/mse_orig)*100,"%")
    # print ("Precision:", ((pre_new-pre_orig)/pre_orig)*100,"%")
    print ("False positive rate:", ((fp_new-fp_orig)/fp_orig)*100,"%")
    print ("Neg Log Loss:", ((log_new-log_orig)/log_orig)*100,"%")