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
    pred.plot(x=feature, y='preds')
    plt.show()

def get_bounds(base, model, feature):
    experiment(base, model, feature)
    bound_i = pd.Series()
    return bound_i