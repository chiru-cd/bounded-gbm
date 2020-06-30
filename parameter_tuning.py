import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

loan = pd.read_csv("loan_2019Q1.csv")
y = loan.pop('class')
X = loan
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42, stratify=y)

params = {
        'base_score': [np.mean(y_train)],
        'eta': [0.1],
        # 'min_child_weight': [1, 5, 10],
        'gamma': [3],
        # 'subsample': [0.8, 1.0],
        'scale_pos_weight': [0.1, 1, 10],
        'max_depth': [5]
        }

xgb = XGBClassifier(n_estimators=100, objective='binary:logistic', silent=True)

folds = 3
param_comb = 10

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='roc_auc', n_jobs=4, cv=skf.split(X_train,y_train), verbose=3, random_state=1001 )
grid = GridSearchCV(estimator=xgb, param_grid=params, scoring='roc_auc', cv=skf.split(X_train,y_train), verbose=3 )

# Here we go
# random_search.fit(X_train, y_train)
grid.fit(X_train, y_train)

print('\n Best estimator:')
# print(random_search.best_estimator_)
print(grid.best_estimator_)

print('\n Best hyperparameters:')
# print(random_search.best_params_)
print(grid.best_params_)

# results = pd.DataFrame(random_search.cv_results_)
# results.to_csv('xgb-random-grid-search-results-01.csv', index=False)

# preds = random_search.predict_proba(X_test)
# preds = grid.best_estimator_.predict_proba(X_test)

# results_df = pd.DataFrame(data={'id':test_df['id'], 'target':y_test[:,1]})
# results_df.to_csv('submission-random-grid-search-xgb-porto-01.csv', index=False)
# print (roc_auc_score(y_test[:,1], preds))
# pickle.dump(model, open("test.dat", "wb"))