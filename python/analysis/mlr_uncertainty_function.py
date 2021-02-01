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


def mlr_uncertainty(stdev_number,location):
    
    sensor = location
    
    if stdev_number == 1 and sensor['Location'].str.contains('Audubon').any():
        sigma_i_summer = 0.78
        slope_sigma_summer = 0.7
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
     
    elif stdev_number == 2 and sensor['Location'].str.contains('Audubon').any():
        sigma_i_summer = 1.56
        slope_sigma_summer = 1.4
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Balboa').any():
        sigma_i_summer = 0.8
        slope_sigma_summer = 0.7
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
      
    elif stdev_number == 2 and sensor['Location'].str.contains('Balboa').any():
        sigma_i_summer = 1.6
        slope_sigma_summer = 1.4
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Browne').any():
        sigma_i_summer = 0.78
        slope_sigma_summer = 0.7
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Browne').any():
        sigma_i_summer = 1.55
        slope_sigma_summer = 1.4
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Lidgerwood').any():
        sigma_i_summer = 1.2
        slope_sigma_summer = 1.1
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Lidgerwood').any():
        sigma_i_summer = 2.4
        slope_sigma_summer = 2.2
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Regal').any():
        sigma_i_summer = 0.87
        slope_sigma_summer = 0.8
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Regal').any():
        sigma_i_summer = 1.74
        slope_sigma_summer = 1.6
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Adams').any():
        sigma_i_summer = 1.82
        slope_sigma_summer = 1.5
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
       
    elif stdev_number == 2 and sensor['Location'].str.contains('Adams').any():
        sigma_i_summer = 3.63
        slope_sigma_summer = 2.9
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Grant').any():
        sigma_i_summer = 1.52
        slope_sigma_summer = 1.2
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Grant').any():
        sigma_i_summer = 3.05
        slope_sigma_summer = 2.4
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Jefferson').any():
        sigma_i_summer = 1.79
        slope_sigma_summer = 1.4
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Jefferson').any():
        sigma_i_summer = 3.58
        slope_sigma_summer = 2.9
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Sheridan').any():
        sigma_i_summer = 1.78
        slope_sigma_summer = 1.4
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Sheridan').any():
        sigma_i_summer = 3.56
        slope_sigma_summer = 2.9
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta =2.1
        sigma_i_BAM = 3.48
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Stevens').any():
        sigma_i_summer = 1.73
        slope_sigma_summer = 1.4
        sigma_i_Augusta = 2.44
        slope_sigma_Augusta = 1
        sigma_i_BAM = 1.74
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Stevens').any():
        sigma_i_summer = 3.47
        slope_sigma_summer = 2.8
        sigma_i_Augusta = 4.89
        slope_sigma_Augusta = 2.1
        sigma_i_BAM = 3.48
        
    
    
    
    sensor['lower_uncertainty'] = sensor['PM2_5_corrected']-(sensor['PM2_5_corrected']*((((((sigma_i_summer/sensor['PM2_5_corrected'])*100))**2+(((sigma_i_Augusta/sensor['PM2_5_corrected'])*100))**2+(((sigma_i_BAM/sensor['PM2_5_corrected'])*100))**2)**0.5)/100))
    sensor['upper_uncertainty'] = sensor['PM2_5_corrected']+(sensor['PM2_5_corrected']*((((((sigma_i_summer/sensor['PM2_5_corrected'])*100))**2+(((sigma_i_Augusta/sensor['PM2_5_corrected'])*100))**2+(((sigma_i_BAM/sensor['PM2_5_corrected'])*100))**2)**0.5)/100))

# Used when had slope uncertainty in the slope calculation, however, just using the residual uncertainites of each of the two steps, and the uncertainty in the BAM
    
#    sensor['lower_uncertainty'] = sensor['PM2_5_corrected']-(sensor['PM2_5_corrected']*(((((((sigma_i_summer/sensor['PM2_5_corrected'])*100))+(((sigma_i_Augusta/sensor['PM2_5_corrected'])*100))**2+slope_sigma_summer**2+slope_sigma_Augusta**2))**0.5)/100))
#    sensor['upper_uncertainty'] = sensor['PM2_5_corrected']+(sensor['PM2_5_corrected']*(((((((sigma_i_summer/sensor['PM2_5_corrected'])*100))+(((sigma_i_Augusta/sensor['PM2_5_corrected'])*100))**2+slope_sigma_summer**2+slope_sigma_Augusta**2))**0.5)/100))

    location = sensor
    
    return location