#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:26:38 2023

@author: laxman
"""
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pandas as pd
data = pd.read_csv('heusler.csv')


label_encoder = LabelEncoder()
encoded_categories = label_encoder.fit_transform(data["heusler type"])
encoded_categories_str_type = label_encoder.fit_transform(data["struct type"])
data["heusler type"] = encoded_categories
data["struct type"] = encoded_categories_str_type
data = data.dropna()  
#print(data.shape)
data=data[data["e_form"]<0] 
data = data[["num_electron","struct type","latt const","heusler type","pol fermi",
          "mu_b","mu_b saturation","e_form"]]

#scaler = StandardScaler()
#data =pd.DataFrame((scaler.fit_transform(data)),columns=data.columns)
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
    
def data_out():
    for col in data.columns:
        data_out = remove_outlier(data, col)

    X = data_out[["num_electron","struct type","latt const","heusler type","pol fermi","mu_b","mu_b saturation"]]
    y = data_out[["e_form"]]
    return X,y 



