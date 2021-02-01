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
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+2.27)/2.35   
    
    else:
            pass
        
    if name == 'Adams': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+1.92)/2.51
    
    else:
            pass

    if name == 'Balboa': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected-2.33)/2.52
    
    else:
            pass      
        
    if name == 'Browne': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+3.31)/2.61
    
    else:
            pass 

    if name == 'Grant': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+2.87)/2.66
    
    else:
            pass

    if name == 'Jefferson': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+2.29)/2.26
    
    else:
            pass

    if name == 'Lidgerwood': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+2.55)/2.61
    
    else:
            pass

    if name == 'Regal': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+2.22)/2.45
    
    else:
            pass

    if name == 'Sheridan': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+2.33)/2.56
    
    else:
            pass

    if name == 'Stevens': 
        indoor_cut['PM2_5_corrected'] = (indoor_cut.PM2_5_corrected+2.8)/2.48
    
    else:
            pass



    return indoor_cut































