#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:48:34 2023

@author: laxman
"""

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dielectric import data_out

X,y = data_out()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=53)

# Define the hyperparameter grid for Grid Search
param_grid = {
    'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],  # Loss function
    'penalty': ['l1', 'l2', 'elasticnet'],  # Regularization penalty
    'alpha': [0.0001, 0.001, 0.01],  # Regularization strength
    'max_iter': [1000, 2000, 3000],  # Maximum number of iterations
}

# Create the SGDRegressor model
model = SGDRegressor(random_state=42)

# Perform Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Train the model with the best hyperparameters
best_model = SGDRegressor(random_state=42, **best_params)
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
