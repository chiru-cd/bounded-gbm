# Currently, the DMLC data parser cannot parse CSV files with headers. Use Pandas to
# read CSV files with headers.
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import operator
import json
import sys
import configparser
from sklearn.metrics import roc_auc_score

from feature_analysis.initialisation import multivar, createBase
from feature_analysis.simulation import simulate
from feature_analysis.experimentation import get_bounds
import feature_analysis.evaluation as ev

config = configparser.ConfigParser()
config.read('configuration.ini')

# loading model
try:
    model_in = config['USER'].get('model')

    if model_in.endswith('.dat'):
        model = pickle.load(open(model_in, "rb"))
    elif model_in.endswith('.model'):
        model = xgb.Booster({'nthread': 4})
        model.load_model(model_in)
except:
    sys.exit("Incorrect model file path or extension")

# input training data
try:
    train_in = config['USER'].get('train')
    train = pd.read_csv(train_in)
except:
    sys.exit("Incorrect training data path")

# check if explicit features provided
if config.has_option('USER', 'features')==False:

    # check if explicit feature importance file provided
    if config.has_option('USER', 'feat_imp')==True:
        try:
            with open(config['USER'].get('feat_imp')) as json_file:
                feat_score = json.load(json_file)
        except:
            sys.exit("Incorrect feature importance file path")
    else:
        try:
            feat_score = model.get_fscore()
        except:
            sys.exit("Provide feature importance file")

    # getting 'n' most important features
    feat_count = config['USER'].getint('feat_count')
    featdict = dict(sorted(feat_score.items(), key=operator.itemgetter(1), reverse=True)[:feat_count])

else:
    featdict = [x.strip() for x in config['USER'].get('features').split(',')]
    featdict = [i for i in featdict if i in feat_score]

# check if explicit bounds provided
if config.has_option('USER', 'bounds')==False:
    poslist = []
    neglist = []

    if config.has_option('USER', 'pos_var')==True:
        poslist = [x.strip() for x in config['USER'].get('pos_var').split(',')]
        poslist = [i for i in poslist if i in feat_score]
    
    if config.has_option('USER', 'neg_var')==True:
        neglist = [x.strip() for x in config['USER'].get('neg_var').split(',')]
        neglist = [i for i in neglist if i in feat_score]

    base = multivar(train, poslist, neglist)
    bounds = pd.DataFrame()

    for i in featdict:
        based = simulate(base, train, i)
        bounds[i] = get_bounds(based, model, i, config['USER'].getfloat('gamma'))

    # option to get boundary values
    if config['USER'].getboolean('get_bounds')==True:
        bounds.to_json("bounds.json")

else:
    try:
        bound_in = config['USER'].get('bounds')
        bounds = pd.read_json(bound_in)
    except:
        sys.exit("Incorrect feature boundaries file path")

# check if test dataset is provided
if config.has_option('USER', 'test')==True:
    # input testing data
    try:
        test_in = config['USER'].get('test')
        test = pd.read_csv(test_in)
    except:
        sys.exit("Incorrect testing data path")

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