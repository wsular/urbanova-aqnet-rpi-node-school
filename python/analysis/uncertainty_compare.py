#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:14:34 2020

@author: matthew
"""

import copy
import pandas as pd
import numpy as np

def uncertainty_compare(df, df_dictionary):


    df_dict = copy.deepcopy(df_dictionary)
    print(len(df_dict))

  #  for key in df_dict.items() :
  #      print (key)
    
    if df['Location'].str.contains('Audubon').any():
  #      print(1)
        df_dict.pop('Audubon', None)
        
  #  for key in df_dict.items() :
  #      print (key)
        
    if df['Location'].str.contains('Adams').any():
  #      print(1)
        del df_dict['Adams']
        
    if df['Location'].str.contains('Balboa').any():
  #      print(1)
        del df_dict['Balboa']
        
    if df['Location'].str.contains('Browne').any():
  #      print(1)
        del df_dict['Browne']
        
    if df['Location'].str.contains('Grant').any():
  #      print(1)
        del df_dict['Grant']
        
    if df['Location'].str.contains('Jefferson').any():
  #      print(1)
        del df_dict['Jefferson']
        
    if df['Location'].str.contains('Lidgerwood').any():
  #      print(1)
        del df_dict['Lidgerwood']
        
    if df['Location'].str.contains('Regal').any():
  #      print(1)
        del df_dict['Regal']
        
    if df['Location'].str.contains('Sheridan').any():
  #      print(1)
        del df_dict['Sheridan']
        
    if df['Location'].str.contains('Stevens').any():
  #      print(1)
        del df_dict['Stevens']
    
    filtered_dict = {}
 
    
    combo = []
    number_of_measurements = []
    location_average = []
    location_median = []
    location_stdev = []
    comparison_average = []
    comparison_median = []
    comparison_stdev = []
    location_over_comparison = []
    location_under_comparison = []
    location_avg_sub_compare_avg = []
    
    print(len(df_dict))

    
    for location in df_dict:
    #    print(1)
        #print location, 'corresponds to', df_dict[location]:
     #   print(df.head())
        name = df['Location'].values[0] + '_' + df_dict[location]['Location'].values[0]
     #   print(2)
     #   print(name)
        over = df
     #   print(3)
     #   print(over.head())
        over['location_upper'] = df_dict[location]['upper_uncertainty']
     #   print(4)
     #   print(over.head())
        over['location_lower'] = df_dict[location]['lower_uncertainty']
     #   print(5)
     #   print(over.head())
        over['location_PM2_5_corrected'] = df_dict[location]['PM2_5_corrected']
        over['location2_name'] = df_dict[location]['Location']
        print(6)
        print(over.head())
        over = over[over['PM2_5_corrected'] > 0]
        over = over.dropna()
     #   print(7)
     #   print(over.head())
        over = over[abs(over['PM2_5_corrected'])-abs(over['location_PM2_5_corrected']) > 0]
        over = over[abs(over['PM2_5_corrected'])-abs(over['location_PM2_5_corrected']) > 3]   # added in to remove low values that could have negative uncertainties that affect calculation
     #  print(8)
     #   print(over.head())
        over = over[abs(over['lower_uncertainty'])-abs(over['location_upper']) > 0]
     #   print(9)
     #   print(over.head())
      #  over = over[over['PM2_5_corrected'] > 0]
        #over = over['PM2_5_corrected'] > 0
      #  print(9)
      #  over['check'] = over['PM2_5_corrected'] - over['location_PM2_5_corrected']
      #  print(10)
      #  over = over[over['check'] > 0]
      #  print(11)

        under = df
      #  print(10)
      #  print(under.head())
        under['location_upper'] = df_dict[location]['upper_uncertainty']
      #  print(11)
      #  print(under.head())
        under['location_lower'] = df_dict[location]['lower_uncertainty']
      #  print(12)
       # print(under.head())
        under['location_PM2_5_corrected'] = df_dict[location]['PM2_5_corrected']
        over['location2_name'] = df_dict[location]['Location']
      #  print(13)
      #  print(under.head())
        under = under[under['PM2_5_corrected'] > 0]
        under = under.dropna()
      #  print(14)
      #  print(under.head())
        under = under[abs(under['location_PM2_5_corrected'])-abs(under['PM2_5_corrected']) > 0]
        under = under[abs(under['location_PM2_5_corrected'])-abs(under['PM2_5_corrected']) > 3]
      #  print(15)
      #  print(under.head())
        under = under[abs(under['location_lower'])-abs(under['upper_uncertainty']) > 0]
      #  print(16)
      #  print(under.head())
        
        combined = over.append(under)
        combined = combined.sort_index()
      #  print(type(combined))
        
        filtered_dict[name] = combined
      #  filtered_dict[name] = over
      #  print(17)
        print(len(df_dict))
    
        combo.append(name)
        number_of_measurements.append(len(combined['Location']))
        location_average.append(np.mean(combined['PM2_5_corrected']))
        location_median.append(np.median(combined['PM2_5_corrected']))
        location_stdev.append(np.std(combined['PM2_5_corrected']))
        comparison_average.append(np.mean(combined['location_PM2_5_corrected']))
        comparison_median.append(np.median(combined['location_PM2_5_corrected']))
        comparison_stdev.append(np.std(combined['location_PM2_5_corrected']))
        location_over_comparison.append(len(over['Location']))
        location_under_comparison.append(len(under['Location']))
        location_avg_sub_compare_avg.append(np.mean(combined['PM2_5_corrected']) - np.mean(combined['location_PM2_5_corrected']))
        
    filtered_dict_stats = pd.DataFrame()
    filtered_dict_stats['Combo'] = combo
    filtered_dict_stats['Number_of_Measurements'] = number_of_measurements
    filtered_dict_stats['Location_Average'] = location_average
    filtered_dict_stats['Location_Median'] = location_median
    filtered_dict_stats['Location_Stdev'] = location_stdev
    filtered_dict_stats['Comparison_Average'] = comparison_average
    filtered_dict_stats['Comparison_Median'] = comparison_median
    filtered_dict_stats['Comparison_Stdev'] = comparison_stdev
    filtered_dict_stats['Location_over_Comparison'] = location_over_comparison
    filtered_dict_stats['Location_under_Comparison'] = location_under_comparison
    filtered_dict_stats['Location_avg_sub_Compare_avg'] = location_avg_sub_compare_avg
        
    filtered_dict_stats = filtered_dict_stats.sort_values('Location_avg_sub_Compare_avg')
        
    return filtered_dict,filtered_dict_stats
        
        