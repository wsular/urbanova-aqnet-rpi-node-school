#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:54:54 2021

@author: matthew
"""

import pandas as pd

def in_out_compare_period_4_data_generator(outdoor_all, start_1, stop_1, start_2, stop_2, interval, location, **kwargs):
    
    unit = kwargs.get('unit', None)
    
    if unit == 'indoor':
        outdoor_all_2 = outdoor_all.copy()
    
        #outdoor_all['Datetime'] = pd.to_datetime(outdoor_all['Datetime'])
        #outdoor_all = outdoor_all.sort_values('Datetime')
       # outdoor_all.index = outdoor_all.Datetime
        outdoor_1 = outdoor_all.loc[start_1:stop_1]
        
        outdoor_1 = outdoor_1.resample(interval).mean() 
        outdoor_1 = outdoor_1.dropna()
        outdoor_1['Location'] = location
        
       # outdoor_all_2['Datetime'] = pd.to_datetime(outdoor_all_2['Datetime'])
       # outdoor_all_2 = outdoor_all_2.sort_values('Datetime')
     #   outdoor_all_2.index = outdoor_all_2.Datetime
        outdoor_2 = outdoor_all_2.loc[start_2:stop_2]
        
        outdoor_2 = outdoor_2.resample(interval).mean() 
        outdoor_2 = outdoor_2.dropna()
        outdoor_2['Location'] = location
        
        outdoor = outdoor_1.append(outdoor_2)
        
    else:
        
        outdoor_all_2 = outdoor_all.copy()
        
        
        outdoor_all['time'] = pd.to_datetime(outdoor_all['time'])
        outdoor_all = outdoor_all.sort_values('time')
        outdoor_all.index = outdoor_all.time
        outdoor_1 = outdoor_all.loc[start_1:stop_1]
        
        outdoor_1 = outdoor_1.resample(interval).mean() 
        outdoor_1 = outdoor_1.dropna()
        outdoor_1['Location'] = location
        
        outdoor_all_2['time'] = pd.to_datetime(outdoor_all_2['time'])
        outdoor_all_2 = outdoor_all_2.sort_values('time')
        outdoor_all_2.index = outdoor_all_2.time
        outdoor_2 = outdoor_all_2.loc[start_2:stop_2]
        
        outdoor_2 = outdoor_2.resample(interval).mean() 
        outdoor_2 = outdoor_2.dropna()
        outdoor_2['Location'] = location
        
        outdoor = outdoor_1.append(outdoor_2)
    
    
    
    return outdoor