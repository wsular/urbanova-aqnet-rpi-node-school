#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:20:52 2020

@author: matthew
"""

def unhealthy_for_sensitive_aqi(location):
    
    location = location[location['PM2_5_corrected'] > 35.4]
    location = location[location['PM2_5_corrected'] <= 55.4]
    location['aqi'] = 'Unhealthy_for_Sensitive'
    location['low_breakpoint'] = 35.4
    location['high_breakpoint'] = 55.4
    
    return(location)