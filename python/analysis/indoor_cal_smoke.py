#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 10:41:27 2021

@author: matthew
"""

import numpy as np

def indoor_cal_smoke(indoor, name):
    
    
    
    start_time = '2020-09-11 08:00'
    end_time = '2020-10-22 07:00'

    indoor_cut = indoor.copy()
    indoor_cut = indoor.loc[start_time:end_time]
    
    if name == 'Audubon': 
        print('audubon')
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected-0.6)*(1/0.66), 
          (indoor_cut.PM2_5_corrected-1.49)/2.04)
    
    else:
            pass
        
    if name == 'Adams': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected-9)*(1/0.63), 
          (indoor_cut.PM2_5_corrected-0.97)/1.9)
    
    else:
            pass

    if name == 'Balboa': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected+18.16)*(1/0.83), 
          (indoor_cut.PM2_5_corrected-1.25)/2.02)
    
    else:
            pass      
        
    if name == 'Browne': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected+27.43)*(1/0.94), 
          (indoor_cut.PM2_5_corrected-0.36)/2.09)
    
    else:
            pass 

    if name == 'Grant': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected+7.68)*(1/0.85), 
          (indoor_cut.PM2_5_corrected-0.88)/2.12)
    
    else:
            pass

    if name == 'Jefferson': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected-2.17)*(1/0.71), 
          (indoor_cut.PM2_5_corrected-0.78)/1.84)
    
    else:
            pass

    if name == 'Lidgerwood': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected+16.05)*(1/0.91), 
          (indoor_cut.PM2_5_corrected-1.08)/2.11)
    
    else:
            pass

    if name == 'Regal': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected-4.12)*(1/0.72), 
          (indoor_cut.PM2_5_corrected-1.14)/1.99)
    
    else:
            pass

    if name == 'Sheridan': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected-1.26)*(1/0.62), 
          (indoor_cut.PM2_5_corrected-1.16)/2.07)
    
    else:
            pass

    if name == 'Stevens': 
        indoor_cut['PM2_5_corrected'] = np.where(indoor_cut.PM2_5_corrected > 68, (indoor_cut.PM2_5_corrected-9.03)*(1/0.67), 
          (indoor_cut.PM2_5_corrected-0.62)/2.01)
    
    else:
            pass



    return indoor_cut


