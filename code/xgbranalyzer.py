#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 08:32:57 2023

@author: laxman
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import cross_val_score,RepeatedKFold
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


class XGBRAnalyzer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.models = None
        self.results = None
        self.names = None

    def get_models(self):
        self.models = dict()
        rates = [0.0001, 0.001, 0.01, 0.1, 1.0]
        for r in rates:
            key = '%.4f' % r
            self.models[key] = XGBRegressor(learning_rate=r)  
    def evaluate_model(self, model):
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(model, self.X, self.y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
        return scores

    def run_analysis(self):
        self.get_models()

        self.results, self.names = list(), list()
        for name, model in self.models.items():
            scores = self.evaluate_model(model)
            self.results.append(scores)
            self.names.append(name)
            data = {'Name': [name],
                'Mean Score': [np.mean(scores)],
                'Standard Deviation': [np.std(scores)]}

            df = pd.DataFrame(data)
            print(df)

    def plot_results(self):
        plt.boxplot(self.results, labels=self.names, showmeans=True)
        plt.xlabel('Learning Rate (Eta)')
        plt.ylabel('Negative Mean Squared Error (MSE)')
        plt.title('XGBoost Regressor Performance vs Learning Rate')
        plt.show()
