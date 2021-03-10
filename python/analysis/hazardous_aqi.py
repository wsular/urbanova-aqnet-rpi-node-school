#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:13:25 2021

@author: matthew
"""

def hazardous_aqi(location):
    
    location = location[location['PM2_5_corrected'] > 250.4]
   # location = location[location['PM2_5_corrected'] <= 250.4]
    location['aqi'] = 'Hazardous'
    location['low_breakpoint'] = 150.4
    location['high_breakpoint'] = 500
    
    return(location)