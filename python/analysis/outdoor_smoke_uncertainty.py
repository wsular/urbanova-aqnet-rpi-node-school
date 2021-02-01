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
        uncertainty_low = 2.57
        uncertainty_high = 22.82
        threshold = 51
     
    elif stdev_number == 2 and sensor['Location'].str.contains('Audubon').any():
        uncertainty_low = 5.14
        uncertainty_high = 45.63
        threshold = 51
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Balboa').any():
        uncertainty_low = 2.58
        uncertainty_high = 18.92
        threshold = 58
      
    elif stdev_number == 2 and sensor['Location'].str.contains('Balboa').any():
        uncertainty_low = 5.15
        uncertainty_high = 37.84
        threshold = 58
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Browne').any():
        uncertainty_low = 2.57
        uncertainty_high = 20.71
        threshold = 74
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Browne').any():
        uncertainty_low = 5.14
        uncertainty_high = 41.41
        threshold = 74
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty_low = 2.73
        uncertainty_high = 18.21
        threshold = 66
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty_low = 5.46
        uncertainty_high = 36.43
        threshold = 66
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Regal').any():
        uncertainty_low = 2.60
        uncertainty_high = 19.80
        threshold = 54
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Regal').any():
        uncertainty_low = 5.20
        uncertainty_high = 39.61
        threshold = 54
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Adams').any():
        uncertainty_low = 3.05
        uncertainty_high = 16.37
        threshold = 75
       
    elif stdev_number == 2 and sensor['Location'].str.contains('Adams').any():
        uncertainty_low = 6.10
        uncertainty_high = 32.74
        threshold = 75
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Grant').any():
        uncertainty_low = 2.94
        uncertainty_high = 18.14
        threshold = 77
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Grant').any():
        uncertainty_low = 5.87
        uncertainty_high = 36.27
        threshold = 77
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty_low = 3.03
        uncertainty_high = 16.43
        threshold = 73
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty_low = 6.07
        uncertainty_high = 32.85
        threshold = 73
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty_low = 3.03
        uncertainty_high = 17.83
        threshold = 82
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty_low = 6.06
        uncertainty_high = 35.66
        threshold = 82
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Stevens').any():
        uncertainty_low = 3.00
        uncertainty_high = 24.48
        threshold = 86
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Stevens').any():
        uncertainty_low = 6.00
        uncertainty_high = 48.97
        threshold = 86
        
    
    
    sensor['lower_uncertainty'] = np.where((sensor.PM2_5_corrected > threshold), sensor.PM2_5_corrected - uncertainty_high,   
                                      sensor.PM2_5_corrected - uncertainty_low)
    
    sensor['upper_uncertainty'] = np.where((sensor.PM2_5_corrected > threshold), sensor.PM2_5_corrected + uncertainty_high,  
                                      sensor.PM2_5_corrected + uncertainty_low)
    
    
    
    location = sensor
    
    return location




    
