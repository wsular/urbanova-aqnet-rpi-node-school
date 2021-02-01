#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:10:59 2021

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:04:08 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:36:15 2020

@author: matthew
"""

#### For use with hourly calibration  #######

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#slope_sigma_summer =  percent uncertainty of SRCAA BAM calibration to reference clarity slope
#slope_sigma_Augusta =  percent uncertainty of slope for Augusta roof calibrations of Clarity Reference node
#sigma_i_summer =   uncertainty of slope for  Clarity unit at Paccar roof calibration in ug/m^3
#sigma_i_Augusta = uncertainty of Clarity Reference node measurements from Augusta calibration in ug/m^3
#sigma_i_BAM = uncertainty of the BAM at the Augusta site as recorded for April which was 1 sigma of 1.74 and 2 sigmas as 3.48 (established as the noise during a zero air test)


def outdoor_low_uncertainty(stdev_number,location):
    
    sensor = location
    
    if stdev_number == 1 and sensor['Location'].str.contains('Audubon').any():
        uncertainty = 2.57
     
    elif stdev_number == 2 and sensor['Location'].str.contains('Audubon').any():
        uncertainty = 5.14
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Balboa').any():
        uncertainty = 2.58
      
    elif stdev_number == 2 and sensor['Location'].str.contains('Balboa').any():
        uncertainty = 5.15
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Browne').any():
        uncertainty = 2.57
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Browne').any():
        uncertainty = 5.14
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty = 2.73
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty = 5.46
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Regal').any():
        uncertainty = 2.60
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Regal').any():
        uncertainty = 5.20
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Adams').any():
        uncertainty = 3.05
       
    elif stdev_number == 2 and sensor['Location'].str.contains('Adams').any():
        uncertainty = 6.10
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Grant').any():
        uncertainty = 2.94
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Grant').any():
        uncertainty = 5.87
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty = 3.03
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty = 6.07
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty = 3.03
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty = 6.06
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Stevens').any():
        uncertainty = 3.00
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Stevens').any():
        uncertainty = 6.00
        
    
    
    sensor['lower_uncertainty'] = sensor['PM2_5_corrected']-uncertainty
    sensor['upper_uncertainty'] = sensor['PM2_5_corrected']+uncertainty
    
    
    
    location = sensor
    
    return location