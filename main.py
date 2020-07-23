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
from feature_analysis.multivariate import monotone

config = configparser.ConfigParser()
config.read('configuration.ini')

# loading model
try:
    model_in = config['USER'].get('model')

    print ("loading model...")
    if model_in.endswith('.dat'):
        model = pickle.load(open(model_in, "rb"))
    elif model_in.endswith('.model'):
        model = xgb.Booster({'nthread': 4})
        model.load_model(model_in)
    print ("model loaded")
except:
    sys.exit("Incorrect model file path or extension")

# input training data
try:
    print ("loading training data...")
    train_in = config['USER'].get('train')
    train = pd.read_csv(train_in)
    print ("training data loaded")
except:
    sys.exit("Incorrect training data path")

# check if explicit features provided
if config.has_option('USER', 'features')==False:

    print ("getting important features...")
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
    featdict = [i for i in featdict if i in train]

print ("important features loaded")
# check if explicit bounds provided
if config.has_option('USER', 'bounds')==False:
    print ("calculating boundaries...")
    
    poslist = []
    neglist = []

    if config.has_option('USER', 'pos_var')==True:
        poslist = [x.strip() for x in config['USER'].get('pos_var').split(',')]
        poslist = [i for i in poslist if i in train]
    
    if config.has_option('USER', 'neg_var')==True:
        neglist = [x.strip() for x in config['USER'].get('neg_var').split(',')]
        neglist = [i for i in neglist if i in train]

    bounds = pd.DataFrame()

    if config['USER'].getfloat('delta') >= 0:
        if len(poslist)>0 or len(neglist)>0:
            # try:
            #     bounds = monotone(train, model, poslist, neglist, config['USER'].getfloat('delta'))
            # except:
            #     sys.exit("Error in multivariate")

            bounds = monotone(train, model, poslist, neglist, config['USER'].getfloat('delta'))
        else:
            sys.exit("Provide correct features in pos_var & neg_var OR set delta < 0")
    
    else:
        try:
            base = multivar(train, poslist, neglist)
        except:
            print ("Error in initialisation")

        for i in featdict:
            try:
                based = simulate(base, train, i)
            except:
                sys.exit("Error in simulation")
            try:
                bounds[i] = get_bounds(based, model, i, config['USER'].getfloat('gamma'))
            except:
                sys.exit("Error in experimentation")
    
    # option to get boundary values
    if config['USER'].getboolean('get_bounds')==True:
        bounds.to_json("bounds.json")

else:
    try:
        bound_in = config['USER'].get('bounds')
        bounds = pd.read_json(bound_in)
    except:
        sys.exit("Incorrect feature boundaries file path")

print ("feature boundaries loaded")

# check if test dataset is provided
if config.has_option('USER', 'test')==True:
    print("evaluating...")
    # input testing data
    try:
        test_in = config['USER'].get('test')
        test = pd.read_csv(test_in)
    except:
        sys.exit("Incorrect testing data path")

    eval_flags = [x.strip() for x in config['USER'].get('eval_flags').split(',')]

    for i in eval_flags:
        if i=="eval_bound":
            if config['USER'].getfloat('delta') < 0:
                ev.eval_bound(train, model, bounds).to_excel("Boundary_evaluation.xlsx")
            else:
                print ("Boundary evaluation work in progress")
        elif i=="result":
            ev.get_result(test, bounds).to_csv("result.csv")
        elif i=="reason":
            ev.get_reason_matrix(test, bounds).to_excel("reason.xlsx")
        elif i=="eval_tool":
            ev.evaluate(test, model, bounds)