#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 09:46:46 2021

@author: matthew
"""



def indoor_cal_low(indoor, name):
    
    start_time = '2019-09-01 07:00'
    end_time = '2020-09-11 07:00'
    indoor_cut = indoor.copy()
    indoor_cut = indoor.loc[start_time:end_time]
    
    if name == 'Audubon': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-1.49)/2.04     
    
    else:
            pass
        
    if name == 'Adams': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-0.97)/1.9
    
    else:
            pass

    if name == 'Balboa': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-1.25)/2.02
    
    else:
            pass      
        
    if name == 'Browne': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-0.36)/2.09
    
    else:
            pass 

    if name == 'Grant': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-0.88)/2.12
    
    else:
            pass

    if name == 'Jefferson': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-0.78)/1.84
    
    else:
            pass

    if name == 'Lidgerwood': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-1.08)/2.11
    
    else:
            pass

    if name == 'Regal': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-1.14)/1.99
    
    else:
            pass

    if name == 'Sheridan': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-1.16)/2.07
    
    else:
            pass

    if name == 'Stevens': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-0.62)/2.01
    
    else:
            pass



    return indoor_cut



































