#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 00:14:45 2023

@author: laxman
"""
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from lightgbm import LGBMRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from tensorflow import keras
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from dielectric import data_out
X,y=data_out()

xgbm_hyperparameters = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.1,
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'gamma': 0,
}

xgbm_mae = [] 
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1)
    model1 = XGBRegressor(**xgbm_hyperparameters)
    model1.fit(X_train,y_train)
    predictions1 = model1.predict(X_test)
    mae1 = mean_absolute_error(y_test, predictions1)
    xgbm_mae.append(mae1)
    
sgd_hyperparameters = {
    'alpha':0.001,
    'loss':'epsilon_insensitive',
    'max_iter':1000,
    'penalty':'elasticnet'
    }

sgd_mae =[]
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1)
    model2 = SGDRegressor(**sgd_hyperparameters)
    model2.fit(X_train,y_train)
    predictions2 = model2.predict(X_test)
    mae2 = mean_absolute_error(y_test, predictions2)
    sgd_mae.append(mae2)
    
rf_hyperparameters = {
    'max_depth':None,
    'max_features': 'sqrt',
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators':300
    }

rf_mae=[]
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1)
    model3 = RandomForestRegressor(**rf_hyperparameters)
    model3.fit(X_train,y_train)
    predictions3 = model3.predict(X_test)
    mae3 = mean_absolute_error(y_test, predictions3)
    rf_mae.append(mae3)
    
lgbm_hyperparameters = {
    'learning_rate':0.1,
    'max_depth':6,
    'n_estimators':100,
    'objective':'regression'
    }

lgbm_mae =[]

for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1)
    model4 = LGBMRegressor(**rf_hyperparameters)
    model4.fit(X_train,y_train)
    predictions4 = model4.predict(X_test)
    mae4 = mean_absolute_error(y_test, predictions4)
    lgbm_mae.append(mae4)


ANN_mae =[]
for _ in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model5 = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  
    ])
    model5.compile(optimizer='adam', loss='mean_squared_error')
    model5.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
    predictions5 = model5.predict(X_test)
    mae5 = mean_absolute_error(y_test, predictions5)
    ANN_mae.append(mae5)


# Assuming you have collected MAE values for each model and stored them in a list or Series.
mae_values = [sgd_mae, ANN_mae, xgbm_mae, lgbm_mae, rf_mae]

# Define model names for labeling the x-axis
model_names = ['SGD', 'Neural Network', 'XGBoost', 'LightGBM', 'Random Forest']

dfs = [pd.DataFrame({'Model': [model_names[i]] * len(mae_values[i]), 'MAE': mae_values[i]}) for i in range(len(mae_values))]

# Concatenate the DataFrames
df_mae = pd.concat(dfs, ignore_index=True)

# Create a boxplot using seaborn
plt.figure()
sns.boxplot(x='Model', y='MAE', data=df_mae)
plt.xlabel('Model')
plt.ylabel('Mean Absolute Error (MAE)')
plt.xticks(rotation=0)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.savefig("maes_models.png", dpi=600)
plt.show()
