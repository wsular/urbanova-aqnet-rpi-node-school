#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:22:22 2020

@author: matthew
"""

def Augusta_BAM_uncertainty(stdev_number,location):
    
    sensor = location
    
    if stdev_number == 1 and sensor['Location'].str.contains('Augusta').any():

        sigma_i_BAM = 1.74
        
    if stdev_number == 2 and sensor['Location'].str.contains('Augusta').any():

        sigma_i_BAM = 3.48
        
    sensor['lower_uncertainty'] = sensor['PM2_5_corrected']-sigma_i_BAM
    sensor['upper_uncertainty'] = sensor['PM2_5_corrected']+sigma_i_BAM
    location = sensor
    
    return location