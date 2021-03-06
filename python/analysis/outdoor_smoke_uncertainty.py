#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:29:39 2021

@author: matthew
"""

import numpy as np

def outdoor_smoke_uncertainty(stdev_number,location):
    
    sensor = location
    
    if stdev_number == 1 and sensor['Location'].str.contains('Audubon').any():
        uncertainty_low = 2.61
        uncertainty_high = 22.80
        threshold = 51
     
    elif stdev_number == 2 and sensor['Location'].str.contains('Audubon').any():
        uncertainty_low = 5.22
        uncertainty_high = 45.60
        threshold = 51
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Balboa').any():
        uncertainty_low = 2.03
        uncertainty_high = 18.90
        threshold = 58
      
    elif stdev_number == 2 and sensor['Location'].str.contains('Balboa').any():
        uncertainty_low = 4.07
        uncertainty_high = 37.80
        threshold = 58
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Browne').any():
        uncertainty_low = 2.03
        uncertainty_high = 20.69
        threshold = 74
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Browne').any():
        uncertainty_low = 4.05
        uncertainty_high = 41.38
        threshold = 74
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty_low = 2.22
        uncertainty_high = 18.20
        threshold = 66
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty_low = 4.44
        uncertainty_high = 36.39
        threshold = 66
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Regal').any():
        uncertainty_low = 2.06
        uncertainty_high = 19.79
        threshold = 54
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Regal').any():
        uncertainty_low = 4.12
        uncertainty_high = 39.57
        threshold = 54
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Adams').any():
        uncertainty_low = 2.02
        uncertainty_high = 16.35
        threshold = 75
       
    elif stdev_number == 2 and sensor['Location'].str.contains('Adams').any():
        uncertainty_low = 4.04
        uncertainty_high = 32.70
        threshold = 75
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Grant').any():
        uncertainty_low = 2.47
        uncertainty_high = 18.12
        threshold = 77
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Grant').any():
        uncertainty_low = 4.95
        uncertainty_high = 36.24
        threshold = 77
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty_low = 2.59
        uncertainty_high = 16.41
        threshold = 73
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty_low = 5.18
        uncertainty_high = 32.81
        threshold = 73
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty_low = 2.58
        uncertainty_high = 17.81
        threshold = 82
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty_low = 5.16
        uncertainty_high = 35.62
        threshold = 82
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Stevens').any():
        uncertainty_low = 2.55
        uncertainty_high = 24.47
        threshold = 86
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Stevens').any():
        uncertainty_low = 5.10
        uncertainty_high = 48.94
        threshold = 86
        
    
    
    sensor['lower_uncertainty'] = np.where((sensor.PM2_5_corrected > threshold), sensor.PM2_5_corrected - uncertainty_high,   
                                      sensor.PM2_5_corrected - uncertainty_low)
    
    sensor['upper_uncertainty'] = np.where((sensor.PM2_5_corrected > threshold), sensor.PM2_5_corrected + uncertainty_high,  
                                      sensor.PM2_5_corrected + uncertainty_low)
    
    
    
    location = sensor
    
    return location




    
