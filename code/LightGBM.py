#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 22:51:05 2023

@author: laxman
"""
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dielectric import data_out

X,y = data_out()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=53)

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
print("R-squared:",r2)

# Feature Importance 
best_model.feature_importances_
imp = best_model.feature_importances_
incl = X.columns.values
idx = np.argsort(imp)[::-1]

# Reduce the figure width and height to make bars thinner
plt.figure(figsize=(8, len(idx) * 0.4))  # Adjust the width and height as needed

sns.barplot(x=imp[idx], y=incl[idx], orient="h", ci=None)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances of LightGBM")
plt.tight_layout()
plt.savefig("lgbm_feature_imp.png",dpi=600)
plt.show()

# MAE VS n_estimator 
mae_values =[]
estimator_range = range(1, 130,5)  # Adjust the range as needed

for n_estimators in estimator_range:
    model = LGBMRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mae_values.append(mae)

plt.figure(figsize=(10, 6))
plt.plot(estimator_range, mae_values, linestyle='-', color='b')
plt.title('MAE vs. n_estimators for LightGBM')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('Mean Absolute Error (MAE)')
plt.savefig("maevsn_est.png", dpi = 600)
plt.show()

#r2 vs n_estimator 
r2_values =[]
estimator_range = range(1, 130,5)  # Adjust the range as needed

for n_estimators in estimator_range:
    model = LGBMRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_values.append(r2)

plt.figure(figsize=(10, 6))
plt.plot(estimator_range, r2_values, linestyle='-', color='r')
plt.title('r2_scores vs. n_estimators for LightGBM')
plt.xlabel('Number of Estimators (n_estimators)')
plt.ylabel('r2_scores')
plt.savefig("lgbm_r2_est.png", dpi = 600)
plt.show()

# prediction vs actual 
values = [i for i in range(98)]
plt.figure()
plt.plot(values,y_test,color="b",label ="Actual",linewidth=2.0)
plt.plot(values,predictions,color='r',label="predicted",linewidth=2.0)
plt.xlabel("Number of Experiment")
plt.ylabel("Band Gap")
plt.title("Actual vs predicted outputs of LightGBM")
plt.legend(loc='best')
plt.savefig("lgbm_predition.png", dpi=600)
plt.show()

