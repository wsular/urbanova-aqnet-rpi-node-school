#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:22:11 2020

@author: matthew
"""

def unhealthy_aqi(location):
    
    location = location[location['PM2_5_corrected'] > 55.4]
    location = location[location['PM2_5_corrected'] <= 150.4]
    location['aqi'] = 'Unhealthy'
    location['low_breakpoint'] = 55.4
    location['high_breakpoint'] = 150.4
    
    return(location)