import pandas as pd
import numpy as np

def createBase (train):
    train_copy = train.copy()
    if 'class' in train_copy:
        train_copy.pop('class')
    # meani = train_copy.mean(skipna = True) 
    mediani = train_copy.median(skipna = True) 
    # based = pd.DataFrame([meani])
    based = pd.DataFrame([mediani])
    based1 = pd.concat([based]*50000, ignore_index=True)

    return based1

def multivar (train, poslist, neglist):
    base = createBase(train)
    columns = list(base)
    
    for i in poslist:
        min_val = train[i].min()
        max_val = train[i].max()
        serpos = train.sample(n = 25000, replace=True)[i]
        low = pd.Series (np.random.randint((2*min_val)-max_val, min_val, size=12500))
        high = pd.Series (np.random.randint(max_val, (2*max_val)-min_val, size=12500))
        serpos = serpos.append([low, high], ignore_index = True)
    
        serpos.sort_values(inplace = True, ignore_index = True)
        base[i] = serpos

    for j in neglist:
        min_val = train[j].min()
        max_val = train[j].max()
        serneg = train.sample(n = 25000, replace=True)[j]
        low = pd.Series (np.random.randint((2*min_val)-max_val, min_val, size=12500))
        high = pd.Series (np.random.randint(max_val, (2*max_val)-min_val, size=12500))
        serneg = serneg.append([low, high], ignore_index = True)
    
        serneg.sort_values(ascending = False, inplace = True, ignore_index = True)
        base[j] = serneg

    return base