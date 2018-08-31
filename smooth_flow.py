# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 12:59:40 2018

@author: Mkhuphuli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#setting working directory to path of test and train data
import os 
#os.chdir("path of test and train data")


def label_encode(data, cols):
    """Converts labels to numerical"""
    from sklearn.preprocessing import LabelEncoder
    Lb_encoder = LabelEncoder()
    for col in cols:
        data[col] =  Lb_encoder.fit_transform(data[col])
    return(data)

def one_hot_encode(data,cols):
    """Converts nominal categorical variables to dummy variables
    Used after applying label_encoder 
    Uses OneHotEncoder"""
    from sklearn.preprocessing import OneHotEncoder
    for col in cols:
        one_hot_encoder = OneHotEncoder(categorical_features = [col])
        data = one_hot_encoder.fit_transform(data)
    return(data)
        
    

