#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:12:55 2020

@author: matthew
"""




def good_aqi(location):
    
    location = location[location['PM2_5_corrected'] <= 12]
    location['aqi'] = 'Good'
    location['low_breakpoint'] = 0
    location['high_breakpoint'] = 12
    
    return(location)