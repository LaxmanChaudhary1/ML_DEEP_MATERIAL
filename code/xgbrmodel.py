#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 08:40:05 2023

@author: laxman
"""
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from dielectric import data_out

X, y = data_out()
print("No of Feature Data Points:",X.size)
hyperparameters = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'gamma': 0,
   # 'min_child_weight': 1,
   # 'reg_alpha': 0.1,
   # 'reg_lambda': 0.1
}

_mae = [] 
_mse = []
_r2 =[]
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1)
    model = XGBRegressor(**hyperparameters)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test,predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    _mse.append(mse)
    _mae.append(mae)
    _r2.append(r2)

mse_mean = np.mean(_mse)    
mse_std = np.std(_mse)
mae_mean = np.mean(_mae)
mae_std = np.std(_mae)
r2_scor = np.mean(_r2)

print(f"Mean absolute error {mae_mean} +- {mae_std}")
print(f"Mean squared error {mse_mean} +- {mse_std}")
print("r2_score:",r2_scor)

X.size+y.size
X.shape

df_test = pd.concat([X_test, y_test], axis=1)

df_test["band_gap predicted"] = predictions
#df_test["error"] = abs(df_test["e_form"] - df_test["e_form predicted"])/abs(df_test["e_form"])
df_test["SE"] = np.std(df_test["band_gap predicted"])/(len(df_test["band_gap predicted"])-1)
print(df_test)
