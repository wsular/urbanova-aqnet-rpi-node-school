#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:27:14 2020

@author: matthew
"""
import pandas as pd
from glob import glob
#%%

def load_indoor(name, df_csv, df_json, interval, **kwargs):
    
    
    
    files_csv   = glob('/Users/matthew/work/data/urbanova/ramboll/' + name + '/BME*.csv')
    files_csv.sort()
    #print(files_csv)
    for file in files_csv:
        df_csv = pd.concat([df_csv, pd.read_csv(file)], sort=False)
    
    df_csv['Datetime'] = pd.to_datetime(df_csv['Datetime'])
    df_csv = df_csv.sort_values('Datetime')
    df_csv.index = df_csv.Datetime
    df_csv = df_csv.resample(interval).mean()
    #df_csv = df_csv.loc[start_time:end_time]
    
    print(df_csv.head)
    
    files_json   = glob('/Users/matthew/work/data/urbanova/ramboll/' + name + '/WSU*.json')
    files_json.sort()
    for file in files_json:
        df_json = pd.concat([df_json, pd.read_json(file)], sort=False)
   # print(files_json)
    
    df_json['Datetime'] = pd.to_datetime(df_json['Datetime'])
    df_json = df_json.sort_values('Datetime')
    df_json.index = df_json.Datetime
    df_json = df_json.resample(interval).mean()
   # df_json = df_json.loc[start_time:end_time]
    
    time_period_4 = kwargs.get('time_period_4')
    
    if time_period_4 == 'yes':
    
        start_1 = kwargs.get('start_1')
        stop_1 = kwargs.get('stop_1')
        start_2 = kwargs.get('start_2')
        stop_2 = kwargs.get('stop_2')
    
        indoor_all_2 = df_csv.copy()
    
        df_csv = df_csv.loc[start_1:stop_1]
        df_csv = df_csv.resample(interval).mean() 
        df_csv = df_csv.dropna()
        
        
        indoor_all_2 = indoor_all_2.loc[start_2:stop_2]
        indoor_all_2 = indoor_all_2.resample(interval).mean() 
        indoor_all_2 = indoor_all_2.dropna()
        
        df_csv = df_csv.append(indoor_all_2)
    
    elif time_period_4 == 'no':
        
        start = kwargs.get('start')
        print(start)
        
        stop = kwargs.get('stop')
        print(stop)
        
        df_csv = df_csv.loc[start:stop]
        df_csv = df_csv.resample(interval).mean() 
        df_csv = df_csv.dropna()
      
    
    return df_csv, df_json
    