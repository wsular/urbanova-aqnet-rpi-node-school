#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:19:21 2020

@author: matthew
"""

def moderate_aqi(location):
    
    location = location[location['PM2_5_corrected'] > 12]
    location = location[location['PM2_5_corrected'] <= 35.4]
    location['aqi'] = 'Moderate'
    location['low_breakpoint'] = 12.1
    location['high_breakpoint'] = 35.4
    
    return(location)