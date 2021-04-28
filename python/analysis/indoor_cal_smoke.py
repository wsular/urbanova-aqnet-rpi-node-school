#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 10:41:27 2021

@author: matthew
"""

import numpy as np

def indoor_cal_smoke(indoor, name):
    
    
   
    start_time = '2020-09-11 00:00'
    end_time = '2020-09-21 19:00'

    indoor_time_cut = indoor.copy()
    indoor_time_cut = indoor.loc[start_time:end_time]
    
    indoor_below_70 = indoor_time_cut[indoor_time_cut['PM2_5_corrected'] < 70]      # note that even though these columns are labeled 'PM2_5_corrected' they have not been altered from the raw measurements yet. This is just a hold-over from perious scripts
    indoor_above_120 = indoor_time_cut[indoor_time_cut['PM2_5_corrected'] > 120]
    
    indoor_cut = indoor_below_70.append(indoor_above_120)
    indoor_cut = indoor_cut.sort_index()
    
    if name == 'Audubon': 
        print('audubon')
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-34.4)*(1/0.61),    # above threshold adjustment
          (indoor_cut.PM2_5_corrected+2.27)/2.35)                                                                                # below threshold adjustment (same as indoor_cal_low equations))
    
    else:
            pass
        
    if name == 'Adams': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-15.8)*(1/0.65), 
          (indoor_cut.PM2_5_corrected+1.92)/2.51)
    
    else:
            pass

    if name == 'Balboa': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected+11.9)*(1/0.82), 
          (indoor_cut.PM2_5_corrected-2.33)/2.52)
    
    else:
            pass      
        
    if name == 'Browne': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected+4.86)*(1/0.91), 
          (indoor_cut.PM2_5_corrected+3.31)/2.61)
    
    else:
            pass 

    if name == 'Grant': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-25.84)*(1/0.8), 
          (indoor_cut.PM2_5_corrected+2.29)/2.26)
    
    else:
            pass

    if name == 'Jefferson': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-47.01)*(1/0.64), 
          (indoor_cut.PM2_5_corrected-0.78)/1.84)
    
    else:
            pass

    if name == 'Lidgerwood': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-20.45)*(1/0.85), 
          (indoor_cut.PM2_5_corrected+2.55)/2.61)
    
    else:
            pass

    if name == 'Regal': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-38.78)*(1/0.67), 
          (indoor_cut.PM2_5_corrected+2.22)/2.45)
    
    else:
            pass

    if name == 'Sheridan': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-38.32)*(1/0.57), 
          (indoor_cut.PM2_5_corrected+2.33)/2.56)
    
    else:
            pass

    if name == 'Stevens': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 70, (indoor_cut.PM2_5_corrected-58.07)*(1/0.62), 
          (indoor_cut.PM2_5_corrected+2.8)/2.48)
    
    else:
            pass

    indoor_cut = indoor_cut.dropna()

    return indoor_cut


