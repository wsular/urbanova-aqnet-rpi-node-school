#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 12:24:17 2021

@author: matthew
"""

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


def indoor_low_uncertainty(stdev_number,location):
    
    sensor = location
    
    if stdev_number == 1 and sensor['Location'].str.contains('Audubon').any():
        uncertainty = 2.33
     
    elif stdev_number == 2 and sensor['Location'].str.contains('Audubon').any():
        uncertainty = 4.66
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Balboa').any():
        uncertainty = 2.34
      
    elif stdev_number == 2 and sensor['Location'].str.contains('Balboa').any():
        uncertainty = 4.67
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Browne').any():
        uncertainty = 2.23
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Browne').any():
        uncertainty = 4.47
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty = 2.32
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Lidgerwood').any():
        uncertainty = 4.64
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Regal').any():
        uncertainty = 2.33
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Regal').any():
        uncertainty = 4.66
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Adams').any():
        uncertainty = 2.38
       
    elif stdev_number == 2 and sensor['Location'].str.contains('Adams').any():
        uncertainty = 4.77
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Grant').any():
        uncertainty = 2.25
        
    elif stdev_number == 2 and sensor['Location'].str.contains('Grant').any():
        uncertainty = 4.51
    
    elif stdev_number == 1 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty = 2.31
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Jefferson').any():
        uncertainty = 4.62
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty = 2.42
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Sheridan').any():
        uncertainty = 4.83
        
    elif stdev_number == 1 and sensor['Location'].str.contains('Stevens').any():
        uncertainty = 2.32
    
    elif stdev_number == 2 and sensor['Location'].str.contains('Stevens').any():
        uncertainty = 4.65
        
    
    
    sensor['lower_uncertainty'] = sensor['PM2_5_corrected']-uncertainty
    sensor['upper_uncertainty'] = sensor['PM2_5_corrected']+uncertainty
    
    
    
    location = sensor
    
    return location