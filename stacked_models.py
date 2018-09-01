# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 12:43:34 2018

@author: Mkhuphuli
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

from  civismlext.stacking import StackedRegressor
from  civismlext.nonnegative import NonNegativeLinearRegression

rng = np.random.RandomState(17)
estlist = [
        ("rf",RandomForestRegressor(random_state=rng)),
        ("lr", LinearRegression()),
        #meta-estimator is the last item on the list
        ("metalr",NonNegativeLinearRegression())
        ]

sm = StackedRegressor(estlist, n_jobs=2)
sm.fit(x_train, y_train)
y_pred= sm.predict(x_test)

redictions=pd.DataFrame({"Id":Id_test, "SalePrice":y_pred})
predictions.to_csv(data_dir+"stacking_submission.csv", index=False)
