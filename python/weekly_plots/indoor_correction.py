#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:27:26 2022

@author: matthew
"""


def indoor_correction(indoor, name, correction_interval):
    
    
    indoor_cut = indoor.copy()
    indoor_cut = indoor_cut.resample(correction_interval).mean()
   
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































