#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:27:14 2020

@author: matthew
"""
import pandas as pd
from glob import glob
#%%

def load_indoor(name, df_csv, df_json, interval, start_time, end_time):
    
    files_csv   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/' + name + '/BME*.csv')
    files_csv.sort()
    #print(files_csv)
    for file in files_csv:
        df_csv = pd.concat([df_csv, pd.read_csv(file)], sort=False)
    
    df_csv['Datetime'] = pd.to_datetime(df_csv['Datetime'])
    df_csv = df_csv.sort_values('Datetime')
    df_csv.index = df_csv.Datetime
    df_csv = df_csv.resample(interval).mean()
    #df_csv = df_csv.loc[start_time:end_time]
    
  #  print(df_csv.head)

    files_json   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/' + name + '/WSU*.json')
    files_json.sort()
    for file in files_json:
        df_json = pd.concat([df_json, pd.read_json(file)], sort=False)
   # print(files_json)
    
    df_json['Datetime'] = pd.to_datetime(df_json['Datetime'])
    df_json = df_json.sort_values('Datetime')
    df_json.index = df_json.Datetime
    df_json = df_json.resample(interval).mean()
    df_json = df_json.loc[start_time:end_time]
    
  #  print(df_json.head)
    
    return df_csv, df_json
    