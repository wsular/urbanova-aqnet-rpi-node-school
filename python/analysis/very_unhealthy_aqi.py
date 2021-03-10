#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:12:14 2021

@author: matthew
"""

def very_unhealthy_aqi(location):
    
    location = location[location['PM2_5_corrected'] > 150.4]
    location = location[location['PM2_5_corrected'] <= 250.4]
    location['aqi'] = 'Very_Unhealthy'
    location['low_breakpoint'] = 150.4
    location['high_breakpoint'] = 250.4
    
    return(location)