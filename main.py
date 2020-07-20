# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import operator
import configparser
from sklearn.metrics import roc_auc_score

from feature_analysis.initialisation import multivar, createBase
from feature_analysis.simulation import simulate
from feature_analysis.experimentation import get_bounds
import feature_analysis.evaluation as ev

config = configparser.ConfigParser()

# config['default'] = {
#         "host" : "192.168.1.1",
#         "port" : "22",
#         "username" : "username",
#         "password" : "password"
#     }

# with open('configuration.ini', 'w') as configfile:
#     config.write(configfile)

config.read('configuration.ini')

# input model
model_in = config['USER'].get('model')
# loading model
model = pickle.load(open(model_in, "rb"))

# input training data
train_in = config['USER'].get('train')
# loading train data
train = pd.read_csv(train_in)

# check if explicit features provided
d = model.get_fscore()
if config.has_option('USER', 'features')==False:
    feat_count = config['USER'].getint('feat_count')

    # getting 'n' most important features
    featdict = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True)[:feat_count])

else:
    featdict = [x.strip() for x in config['USER'].get('features').split(',')]
    featdict = [i for i in featdict if i in d]

# check if explicit bounds provided
if config.has_option('USER', 'bounds')==False:
    poslist = []
    neglist = []

    if config.has_option('USER', 'pos_var')==True:
        poslist = [x.strip() for x in config['USER'].get('pos_var').split(',')]
        poslist = [i for i in poslist if i in d]
    
    if config.has_option('USER', 'neg_var')==True:
        neglist = [x.strip() for x in config['USER'].get('neg_var').split(',')]
        neglist = [i for i in neglist if i in d]

    base = multivar(train, poslist, neglist)
    bounds = pd.DataFrame()

    for i in featdict:
        based = simulate(base, train, i)
        bounds[i] = get_bounds(based, model, i)

    # option to get boundary values
    if config['USER'].getboolean('get_bounds')==True:
        bounds.to_json("bounds.json")

else:
    bound_in = config['USER'].get('bounds')
    bounds = pd.read_json(bound_in)

# check if test dataset is provided
if config.has_option('USER', 'test')==True:
    # input testing data
    test_in = config['USER'].get('test')
    # loading test data
    test = pd.read_csv(test_in)

    eval_flags = [x.strip() for x in config['USER'].get('eval_flags').split(',')]

    for i in eval_flags:
        if i=="eval_bound":
            ev.eval_bound(train, model, bounds).to_excel("Boundary_evaluation.xlsx")
        elif i=="result":
            ev.get_result(test, bounds).to_csv("result.csv")
        elif i=="reason":
            ev.get_reason_matrix(test, bounds).to_excel("reason.xlsx")
        elif i=="eval_tool":
            ev.evaluate(test, model, bounds)