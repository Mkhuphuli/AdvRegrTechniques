# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 13:28:06 2018

@author: Mkhuphuli
"""
""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv("data\\train_clean.csv")
test =pd.read_csv("data\\test_clean.csv")


if 'Unnamed: 0' in train.columns:
    train = train.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in test.columns:
    test = test.drop('Unnamed: 0', axis=1)
    
y_train = train["SalePrice"]
x_train = train.drop("SalePrice", axis=1)

concat_row=len(x_train)
all_data=pd.concat([x_train,test], axis=0)

def get_categorial_variables(df):
    """Takes one argument, a pandas dataframe and returns a list
    of categorical varibles"""
    
    cols=[]
    for col in df.columns:
        if df[col].dtype == "O":
            cols.append(col)
    return(cols)


def encode_labels_and_get_dummies(df):
    """Uses the get_categorial_variables(df) function
    to choose the appropriate columns to encode"""

    cols = get_categorial_variables(df)
    
    #Dealing with ordinal variable
    #encode of columns with the same cond/quality ranking style/naming
    qual_encode_dict = {"None":0, "Po":1, "Fa":2, "TA":3,"Gd":4, "Ex":5}
    qual_col = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", 
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
    #convert all cond/quality levels with integers for the columns in qual_col
    df[qual_col]=df[qual_col].replace(qual_encode_dict)
    
    #encoding norminal variables
    df=pd.get_dummies(df, columns=list(set(cols)-set(qual_col)))
           
    
    return(df)

all_data=encode_labels_and_get_dummies(all_data)

x_train = all_data[:concat_row]
test = all_data[concat_row:]


#splitting the dataset into the Training set and Validation set
#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

Id_train=x_train["Id"]
Id_test=test["Id"]# **test dataset has no observed y
x_train=x_train.drop("Id", axis=1)
x_test=test.drop("Id", axis=1)# **test dataset has no observed y

from sklearn.linear_model import LinearRegression as LR
regressor = LR()
regressor.fit(x_train, y_train)

# Predicting the test results
y_pred = regressor.predict(x_test)

predictions=pd.DataFrame({"Id":Id_test, "SalePrice":y_pred})
predictions.to_csv("C:/Users/Mkhuphuli/Documents/MSc/DSI_program/Kaggle/House prices/data/lr_submission.csv", index=False)
