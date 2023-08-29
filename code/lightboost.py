#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 11:05:01 2023

@author: laxman
"""
import pandas as pd
import numpy as np 
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dielectric import data_out

X,y = data_out()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=53)

# Define the hyperparameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosting iterations
    'max_depth': [6, 8, 10],  # Maximum depth of each tree
    'learning_rate': [0.1, 0.01],
    'objective': ['regression'],  # Specify 'regression' for regression tasks
}

# Create the LGBMRegressor model
model = LGBMRegressor()

# Perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_model = LGBMRegressor(**best_params)
best_model.fit(X_train, y_train)

# Make predictions
predictions = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Print evaluation metrics
print("Best Hyperparameters:", best_params)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R-squared:", r2)

df_test = pd.concat([X_test, y_test], axis=1)
df_test["band_gap predicted"] = predictions
#df_test["error"] = abs(df_test["e_form"] - df_test["e_form predicted"])/abs(df_test["e_form"])
df_test["SE"] = np.std(df_test["band_gap predicted"])/(len(df_test["band_gap predicted"])-1)
print(df_test)

n_estimator = range(1,101)
mae_ = []
r2_ = []
for n in n_estimator:
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1,random_state=53)
    model = LGBMRegressor(n_estimators=n,random_state=53)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test,predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae_.append(mae)
    r2_.append(r2)
    
import matplotlib.pyplot as plt 
plt.figure(figsize=(10,6))
plt.plot(n_estimator,mae_,color = 'b',linestyle='-',linewidth=2)
plt.title("MAE vs n_estimators for lgbm.")
plt.xlabel("n_estimators for lgbm.")
plt.ylabel("Mean Absolute Error.")
plt.savefig("lgbm_maevsn_estimator.png",dpi=300)


plt.figure(figsize=(10,6))
plt.plot(n_estimator,r2_,color = 'r',linestyle='-',linewidth=2)
plt.title("r2_score vs n_estimators for lgbm.")
plt.xlabel("n_estimators for xgboost.")
plt.ylabel("r2_score.")
plt.savefig("lgbm_r2vsn_estimator.png",dpi=300)

number = [k for k in range(98)]
ax =plt.gca()
ax.plot(number,df_test['band_gap predicted'],color = 'r',linestyle='-',linewidth=2)
plt.plot(number,df_test['band_gap'])
plt.title("predicted vs actual band_gap for lgbm.")
plt.xlabel("values for lgbm.")
plt.ylabel("predictions")
ax.set_yticks(np.arange(0,1,step =0.2))
plt.savefig("lgbm_bandgap.png",dpi=300)


import seaborn as sns 
model.feature_importances_
imp = model.feature_importances_
incl = X.columns.values
idx = np.argsort(imp)[::-1]

# Reduce the figure width and height to make bars thinner
plt.figure(figsize=(8, len(idx) * 0.5))  # Adjust the width and height as needed

sns.barplot(x=imp[idx], y=incl[idx], orient="h", ci=None)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances of lgbm.")
plt.tight_layout()
plt.savefig("lgbm_featureimp.png",dpi=300)
plt.show()
