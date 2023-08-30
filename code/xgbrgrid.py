#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 10:37:28 2023

@author: laxman
"""
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from  dielectric import data_out

X,y=data_out()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                     test_size=0.1,random_state=53)

n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=53)
xgb = XGBRegressor()

param_grid = {
    'n_estimators':[50, 100, 200,300,350],
    'learning_rate': [0.01, 0.1, 0.2,0.02,0.001],
    'max_depth': [3, 4, 5,6,7,8,9],
    'subsample':[0.9,0.8,0.7,0.6,0.5,0.4,0.3],
    'colsample_bytree':[0.8,0.7,0.6,0.5,0.4,0.3,0.2],
    'gamma':[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
}

grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=kf, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)


print("Best Hyperparameters:", grid_search.best_params_)

# Get the best estimator from GridSearchCV
best_xgb = grid_search.best_estimator_

# Perform k-fold cross-validation and calculate MSE scores using the best estimator
mse_scores = cross_val_score(best_xgb, X, y, cv=kf, scoring='neg_mean_squared_error')

average_mse = -np.mean(mse_scores)
print("Average Mean Squared Error with Best Estimator:", average_mse)

# for prediction 
best_xgb.fit(X_train, y_train)
# using the best estimator
y_pred = best_xgb.predict(X_test)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test, y_pred)
# Calculate Mean Squared Error on the test set
test_mse = mean_squared_error(y_test, y_pred)
print("Test Mean Squared Error with Best Estimator:", test_mse)

df_test = pd.concat([X_test, y_test], axis=1)

df_test["band_gap predicted"] = y_pred
#df_test["error"] = np.std(y_pred)/(len(y_pred)-1)
print(df_test)

best_xgb.feature_importances_
imp = best_xgb.feature_importances_
incl = X.columns.values
idx = np.argsort(imp)[::-1]

# Reduce the figure width and height to make bars thinner
plt.figure(figsize=(8, len(idx) * 0.4))  # Adjust the width and height as needed

sns.barplot(x=imp[idx], y=incl[idx], orient="h", ci=None)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances of XGBM")
plt.tight_layout()
plt.savefig("xgbm_feature.png", dpi = 600)
plt.show()

# MAE VS n_estimator 
mae_values =[]
estimator_range = range(1, 206,3)  # Adjust the range as needed

for n_estimators in estimator_range:
    model = XGBRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_values.append(mae)

plt.figure()
plt.plot(estimator_range, mae_values, linestyle='-', color='b')
plt.title('MAE vs. n_estimators for XGBM')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.savefig("xgbm_mae.png", dpi = 600)
plt.show()

#r2 vs n_estimator 
r2_values =[]
estimator_range = range(1, 206,3)  # Adjust the range as needed

for n_estimators in estimator_range:
    model = XGBRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_values.append(r2)

plt.figure()
plt.plot(estimator_range, r2_values, linestyle='-', color='r')
plt.title('r2_scores vs. n_estimators for XGBM')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('r2_scores')
plt.savefig("xgbm_r2_est.png", dpi = 600)
plt.show()

values = [i for i in range(98)]
plt.figure()
plt.plot(values,y_test,color="b",label ="Actual",linewidth=2.0)
plt.plot(values,y_pred,color='r',label="predicted",linewidth=2.0)
plt.xlabel("Number of Experiment")
plt.ylabel("Band Gap")
plt.title("Actual vs predicted outputs of XGBM")
plt.legend()
plt.savefig("xgbm_predition.png", dpi=600)
plt.show()

