#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:47:59 2021

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 10:41:27 2021

@author: matthew
"""
from high_cal_mlr_function import mlr_function_high_cal
import numpy as np

def outdoor_cal_smoke(outdoor, name, high_mlr_function):
    
    
    
    start_time = '2020-09-11 08:00'
    end_time = '2020-10-22 07:00'

    outdoor_cut = outdoor.copy()
    outdoor_cut = outdoor.loc[start_time:end_time]
    
    if name == 'Audubon': 
        print('audubon')
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 51), mlr_function_high_cal(high_mlr_function, outdoor_cut),  # high calibration adjustment
                                       ((outdoor_cut.PM2_5-0.4207)/1.0739)                               # Paccar roof adjustment
                                       *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)   # high calibration adjustment
    
    else:
            pass
        
    if name == 'Adams': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 75), mlr_function_high_cal(high_mlr_function, outdoor_cut),  # high calibration adjustment
                                    ((outdoor_cut.PM2_5+0.93)/1.1554)                             # Paccar roof adjustment
                                    *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)   # high calibration adjustment
    
    else:
            pass

    if name == 'Balboa': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 58), mlr_function_high_cal(high_mlr_function, outdoor_cut), 
                                     ((outdoor_cut.PM2_5-0.2878)/1.2457)  #  Paccar roof adjustment
                                     *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)  # high calibration adjustment
    
    else:
            pass      
        
    if name == 'Browne': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 74), mlr_function_high_cal(high_mlr_function, outdoor_cut),   # high calibration adjustment
                                     ((outdoor_cut.PM2_5-0.4771)/1.1082)                               # Paccar roof adjustment
                                     *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)    # high calibration adjustment
    
    else:
            pass 

    if name == 'Grant': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 77), mlr_function_high_cal(high_mlr_function, outdoor_cut),  # high calibration adjustment
                                    ((outdoor_cut.PM2_5+1.0965)/1.29)                              # Paccar roof adjustment
                                    *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)  # high calibration adjustment
    
    else:
            pass

    if name == 'Jefferson': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 73), mlr_function_high_cal(high_mlr_function, outdoor_cut),  # high calibration adjustment
                                        ((outdoor_cut.PM2_5+0.7099)/1.1458)                                # Paccar roof adjustment
                                        *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)  # high calibration adjustment
    
    else:
            pass

    if name == 'Lidgerwood': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 66),  mlr_function_high_cal(high_mlr_function, outdoor_cut),   # high calibration adjustment
                                         (outdoor_cut.PM2_5-1.1306)/0.9566                                    # Paccar roof adjustment
                                         *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)  # high calibration adjustment
    
    else:
            pass

    if name == 'Regal': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 54),  mlr_function_high_cal(high_mlr_function, outdoor_cut),   # high calibration adjustment
                                    ((outdoor_cut.PM2_5-0.247)/0.9915)                                    # Paccar roof adjustment
                                    *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)         # high calibration adjustment
    
    else:
            pass

    if name == 'Sheridan': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 82), mlr_function_high_cal(high_mlr_function, outdoor_cut),  # high calibration adjustment
                                       ((outdoor_cut.PM2_5+0.6958)/1.1468)                             # Paccar roof adjustment
                                       *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)  # high calibration adjustment
    
    else:
            pass

    if name == 'Stevens': 
        outdoor_cut['PM2_5_corrected'] = np.where((outdoor_cut.PM2_5 > 86), mlr_function_high_cal(high_mlr_function, outdoor_cut),   # high calibration adjustment
                                      ((outdoor_cut.PM2_5+0.8901)/1.2767)                                 # Paccar roof adjustment
                                      *0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242)     # high calibration adjustment
    
    else:
            pass


    return outdoor_cut


