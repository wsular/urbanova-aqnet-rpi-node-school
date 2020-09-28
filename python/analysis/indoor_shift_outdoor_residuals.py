#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 13:39:50 2020

@author: matthew
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def in_out_histogram(indoor,outdoor, df_list):
    
    copy_indoor = indoor
    
    copy_indoor['Outdoor Calibrated'] = outdoor['PM2_5_corrected']
    copy_indoor['Shifted Residuals'] = copy_indoor['Outdoor Calibrated'] - copy_indoor['PM2_5_corrected_shift']
    copy_indoor['Fraction Infiltrated'] = copy_indoor['PM2_5_corrected_shift']/copy_indoor['Outdoor Calibrated']
    copy_indoor['Fraction Filtered'] = (copy_indoor['Outdoor Calibrated']-copy_indoor['PM2_5_corrected_shift'])/copy_indoor['Outdoor Calibrated']
    copy_indoor = copy_indoor.dropna()
    
    total_measurements = len(copy_indoor)
    print(total_measurements)
  #  residuals = indoor['Shifted_Residuals']
    
    column_list = ['Shifted Residuals', 'Fraction Infiltrated', 'Fraction Filtered']
    
    # Number of times the threshold is applied (limit to 2 for Audubon and Adams)
    thresholds = [0,1,2,3]
  #  df_list = [df_threshold_1, df_threshold_2, df_threshold_3, df_threshold_4]
    
    # Value that is multiplied by threshold to divide main df
    interval = 5
    
    location_name = indoor.iloc[0]['Location']
    
    dummy_df_list = []
    #print(location_name)
    
    # This should end up creating df's of all data for when the Outdoor column is > 5 ug/m3 on the first loop, then > 10 on the second, then > 15 on the third loop
    for threshold, df in zip(thresholds,df_list):
      #  print('start')
      #  print(threshold, df)
        
        lower_limit = threshold*interval
     #   print(lower_limit)
        
        df_cut = copy_indoor[copy_indoor['Outdoor Calibrated'] > lower_limit]
        print(df_cut.head())
    
        df_location = df['Location'].tolist()
        df_avg_fraction_filtered = df['avg_fraction_filtered'].tolist()
        df_out_avg = df['out_avg'].tolist()
        df_in_avg = df['in_avg'].tolist()
        df_out_med = df['out_med'].tolist()
        df_in_med = df['in_med'].tolist()
        df_count = df['count'].tolist()
        df_percentage = df['percentage_total_measurements'].tolist()  ###
        print(df_percentage)                                           ###
        df_location.append(location_name)
        df_avg_fraction_filtered.append(np.mean(df_cut['Fraction Filtered']))
        df_out_avg.append(np.mean(df_cut['Outdoor Calibrated']))
        df_in_avg.append(np.mean(df_cut['PM2_5_corrected_shift']))
        df_out_med.append(np.nanmedian(df_cut['Outdoor Calibrated']))
        df_in_med.append(np.nanmedian(df_cut['PM2_5_corrected_shift']))
        df_count.append(len(df_cut))
        print(len(df_cut))
        print(total_measurements)
        number_measurements = len(df_cut)
        print(number_measurements)
        df_percentage.append((number_measurements/total_measurements)*100)     ###
        
       # l_df_location,l_df_avg_fraction_filtered,l_df_out_avg, l_df_in_avg, l_df_out_med, l_df_in_med = len(df_location),len(df_avg_fraction_filtered),len(df_out_avg),len(df_in_avg),len(df_out_med),len(df_in_med)

#        if not number_locations == l_df_location:
#            df_avg_fraction_filtered.extend(['']*(number_locations-l_df_location))
#        if not number_locations == l_df_avg_fraction_filtered:
#            df_out_avg.extend(['']*(number_locations-l_df_avg_fraction_filtered))
#        if not number_locations == l_df_out_avg:
#            df_in_avg.extend(['']*(number_locations-l_df_out_avg))
            
#        if not number_locations == l_df_in_avg:
#            df_in_avg.extend(['']*(number_locations-l_df_in_avg))
#        if not number_locations == l_df_out_med:
#            df_out_med.extend(['']*(number_locations-l_df_out_med))
#        if not number_locations == l_df_in_med:
#            df_in_med.extend(['']*(number_locations-l_df_in_med))
        
        #print(df_location)
        
        # Create dummy dataframe to assign lists to
        df1 = pd.DataFrame({})
        
        df1['Location'] = df_location
        df1['avg_fraction_filtered'] = df_avg_fraction_filtered
        df1['out_avg'] = df_out_avg
        df1['in_avg'] = df_in_avg
        df1['out_med'] = df_out_med
        df1['in_med'] = df_in_med
        df1['count'] = df_count
        df1['percentage_total_measurements'] = df_percentage         ###
       # df1['percentage_total_measurements'] = df1['percentage_total_measurements']   # turn from fraction into percentage
        df1 = df1.sort_values('avg_fraction_filtered', ascending=False)
        
     #   print(1)
     #   print(df1)
     #   print(2)
     #   print(df)
        
        # Assign dummy df to the original df input so that the updated df is returned and can be used in the next instance of the function (ie next location)
        df = df1
     #   print(3)
      #  print(df)

        dummy_df_list.append(df)
        
      #  if lower limit ==
    
    
  # #     for i in column_list:
        
       #     print(i)
  ##          residuals = df_cut[i]
         #   print(np.min(residuals))
   ##         std_dev = residuals.std()
    ##        two_std_dev = 2*std_dev
     ##       mean,std=norm.fit(residuals)
            
     ##       figure = plt.figure()
      ##      plot_title = (df_cut.iloc[0]['Location']) + ' ' + i + ' Lower Limit = ' + str(lower_limit)
       ##     plt.xlabel(i, fontsize=14)
        ##    figure.suptitle(plot_title, fontsize = 16)
       ##     plt.ylabel('Count', fontsize=14)
       ##     plt.hist(residuals, bins=30, alpha=0.5, histtype='bar', ec='black')#, density=True)
            # xmin, xmax = plt.xlim()
            # x = np.linspace(xmin, xmax, 100)
            # y = norm.pdf(x, mean, std)
            #plt.plot(x, y)
       ##     plt.show()
    
          #  print(plot_title + ' one sigma ' + '= ' + str(std_dev))
          #  print(plot_title + ' two sigmas ' + '= ' + str(two_std_dev))
                
    return(dummy_df_list)
  #  return(df_threshold_1, df_threshold_2, df_threshold_3, df_threshold_4)