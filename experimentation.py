import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

def experiment(base, model, feature):
    X_test = xgb.DMatrix(data = base)
    pred = pd.DataFrame(columns = [feature, 'preds'])

    pred[feature] = base[feature]
    pred['preds'] = model.predict(X_test)
    pred.sort_values(feature, inplace = True)
    # pred.plot(x=feature, y='preds')
    # plt.show()
    return pred

def get_bounds(base, model, feature):
    df = experiment(base, model, feature)
    df['preds_diff'] = df['preds'].diff()
    indices = df.index[df['preds_diff'] != 0].tolist()
    lowi = indices[1]
    highi = indices[-1]
    low = df.at[lowi, feature]
    high = df.at[highi, feature]
    bound_i = pd.Series([low, high])
    return bound_i