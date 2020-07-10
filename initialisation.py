import pandas as pd
import numpy as np

def createBase (data_in):
    data_in.pop('class')
    meani = data_in.mean(skipna = True) 
    based = pd.DataFrame([meani])
    based1 = pd.concat([based]*50000, ignore_index=True)

    return based1