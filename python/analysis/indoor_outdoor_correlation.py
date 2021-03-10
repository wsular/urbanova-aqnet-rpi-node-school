#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:47:03 2020

@author: matthew
"""
import copy

def in_out_corr(corr_df, indoor, outdoor):
    
    indoor = copy.deepcopy(indoor)
    
    location_name = indoor.iloc[0]['Location']
    shifts = [0, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            #  -1,-1,-1,-1,-1]
 #   offsets = [0, -1, -2, -3, -4, -5]
    
    corr_list = []
    corr_squared_list = []
    
    indoor['out_PM2_5_corrected'] = outdoor['PM2_5_corrected']

    for hour in shifts:
     #   indoor['Datetime'] = pd.DatetimeIndex(indoor['Datetime']) + timedelta(minutes=offset)
        indoor_shift = indoor
        indoor_shift['PM2_5_corrected'] = indoor_shift['PM2_5_corrected'].shift(hour)
        corr = indoor_shift['PM2_5_corrected'].corr(indoor_shift['out_PM2_5_corrected'])
        corr_squared = corr**2
        corr_list.append(corr)
        corr_squared_list.append(corr_squared)
       # print(corr_squared)
        
    
    corr_df[location_name] = corr_list
    #corr_df[location_name] = corr_squared_list
    #df['location'] = location_name
    #df['offset'] = offsets
    
    return corr_df
    