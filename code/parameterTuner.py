#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:11:27 2023

@author: laxman
"""

from xgbrn_estimatoranalyzer import XGBn_estimatorAnalyzer
from xgbranalyzer import XGBRAnalyzer
from xgbrcolumnratioanalyzer import XGBRColumnRatioAnalyzer
from xgbrsubsampleanalyzer import XGBRSubsampleAnalyzer
from xgbrtreedepthanalyzer import XGBRTreeDepthAnalyzer
from processeddata import data_out 
from sklearn.model_selection import train_test_split

X,y = data_out()
X_train,X_test,y_train,t_test = train_test_split(X,y,test_size=0.1)

analyzer1 = XGBn_estimatorAnalyzer(X_train, y_train)
analyzer1.run_analysis()
analyzer1.plot_results()

analyzer2 = XGBRTreeDepthAnalyzer(X, y)
analyzer2.run_analysis()
analyzer2.plot_results()

analyzer3 = XGBRAnalyzer(X, y)
analyzer3.run_analysis()
analyzer3.plot_results()

analyzer4 = XGBRSubsampleAnalyzer(X, y)
analyzer4.run_analysis()
analyzer4.plot_results()

analyzer5 = XGBRColumnRatioAnalyzer(X, y)
analyzer5.run_analysis()
analyzer5.plot_results()