import pandas as pd
import numpy as np

def createBase (data_in):
    train_copy = data_in.copy()
    train_copy.pop('class')
    # meani = train_copy.mean(skipna = True) 
    mediani = train_copy.median(skipna = True) 
    # based = pd.DataFrame([meani])
    based = pd.DataFrame([mediani])
    based1 = pd.concat([based]*50000, ignore_index=True)

    return based1