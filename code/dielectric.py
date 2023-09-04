#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 18:51:16 2023

@author: laxman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 09:26:38 2023

@author: laxman
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
data = pd.read_csv("../data/dielectric_materials.csv")
data["poly_ionic"] =data["poly_total"] - data["poly_electronic"]
data = data.dropna()  
data = data.drop(["material_id","formula","structure","cif","meta","poscar","pot_ferroelectric","e_electronic","e_total"],axis=1)

scaler = MinMaxScaler()
data =pd.DataFrame((scaler.fit_transform(data)),columns=data.columns)
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

    X = data_out[['nsites', 'space_group', 'volume', 'n', 'poly_electronic',
           'poly_total', 'poly_ionic']]
    y = data_out[['band_gap']]
    return X,y 



