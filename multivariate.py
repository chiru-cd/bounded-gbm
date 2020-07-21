import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import operator
import configparser
from sklearn.metrics import roc_auc_score

config = configparser.ConfigParser()
config.read('configuration.ini')

model_in = config['USER'].get('model')
if model_in.endswith('.dat'):
    model = pickle.load(open(model_in, "rb"))
elif model_in.endswith('.model'):
    model = xgb.Booster({'nthread': 4})
    model.load_model(model_in)

train_in = config['USER'].get('train')
train = pd.read_csv(train_in)

feat_score = model.get_fscore()
pos_list = [x.strip() for x in config['USER'].get('pos_var').split(',')]
pos_list = [i for i in pos_list if i in feat_score]
neg_list = [x.strip() for x in config['USER'].get('neg_var').split(',')]
neg_list = [i for i in neg_list if i in feat_score]

df = pd.DataFrame()

a = [0] * 30
for j in range (0, 30):
    train_copy = train.sample(n = 25000)

    for i in pos_list:
        min_val = train[i].min()
        max_val = train[i].max()
        diff = (max_val - min_val)
        new_val = min_val + ((j/10)-1)*diff
        if 'fico' in i:
            if new_val < 0:
                new_val = 0
            if new_val > 900:
                new_val = 900
        train_copy[i] = new_val

    for i in neg_list:
        min_val = train[i].min()
        max_val = train[i].max()
        diff = (max_val - min_val)
        new_val = max_val - ((j/10)-1)*diff
        if 'fico' in i:
            if new_val < 0:
                new_val = 0
            if new_val > 900:
                new_val = 900
        train_copy[i] = new_val

    ylt = train_copy.pop('class')

    # creating Dmatrix of the modified data
    X_test_lt = xgb.DMatrix(data=train_copy)

    # performing prediction on new test dataset
    predslt =  model.predict(X_test_lt)

    a[j] = np.mean(predslt)

df["mean_preds"] = pd.Series(a)
df.plot()
plt.show()