#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:39:41 2021

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:29:39 2021

@author: matthew
"""

import numpy as np

def indoor_smoke_uncertainty(stdev_number,location):
    
    sensor = location
    
    if stdev_number == 1 and sensor['Location'].str.contains('Audubon').any():
        uncertainty_low = 2.33
        uncertainty_high = 113.41
        threshold = 51
     
    elif stdev_number == 2 and sensor['Location'].str.contains('Audubon').any():
        uncertainty_low = 4.66
        uncertainty_high = 226.82
        threshold = 51
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Balboa').any():
        uncertainty_low = 2.34
        uncertainty_high = 109
        threshold = 58
      
    elif stdev_number == 2 and sensor['Location'].str.contains('Balboa').any():
        uncertainty_low = 4.67
        uncertainty_high = 218
        threshold = 58
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Browne').any():
        uncertainty_low = 2.23
        uncertainty_high = 103
        threshold = 74
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Browne').any():
        uncertainty_low = 4.47
        uncertainty_high = 206
        threshold = 74
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty_low = 2.32
        uncertainty_high = 105
        threshold = 66
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty_low = 4.64
        uncertainty_high = 210
        threshold = 66
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Regal').any():
        uncertainty_low = 2.33
        uncertainty_high = 126
        threshold = 54
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Regal').any():
        uncertainty_low = 4.66
        uncertainty_high = 253
        threshold = 54
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Adams').any():
        uncertainty_low = 2.38
        uncertainty_high = 146
        threshold = 75
       
    elif stdev_number == 2 and sensor['Location'].str.contains('Adams').any():
        uncertainty_low = 4.77
        uncertainty_high = 293
        threshold = 75
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Grant').any():
        uncertainty_low = 2.25
        uncertainty_high = 102
        threshold = 77
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Grant').any():
        uncertainty_low = 4.51
        uncertainty_high = 205
        threshold = 77
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty_low = 2.31
        uncertainty_high = 118
        threshold = 73
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty_low = 4.62
        uncertainty_high = 237
        threshold = 73
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty_low = 2.42
        uncertainty_high = 207
        threshold = 82
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty_low = 4.83
        uncertainty_high = 415
        threshold = 82
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Stevens').any():
        uncertainty_low = 2.32
        uncertainty_high = 145
        threshold = 86
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Stevens').any():
        uncertainty_low = 4.65
        uncertainty_high = 291
        threshold = 86
        
    
    
    sensor['lower_uncertainty'] = np.where((sensor.PM2_5_corrected > threshold), sensor.PM2_5_corrected - uncertainty_high,   
                                      sensor.PM2_5_corrected - uncertainty_low)
    
    sensor['upper_uncertainty'] = np.where((sensor.PM2_5_corrected > threshold), sensor.PM2_5_corrected + uncertainty_high,  
                                      sensor.PM2_5_corrected + uncertainty_low)
    
    
    
    location = sensor
    
    return location




    
