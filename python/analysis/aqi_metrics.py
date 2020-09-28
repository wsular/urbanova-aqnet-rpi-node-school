#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 13:59:23 2020

@author: matthew
"""

import pandas as pd
import numpy as np

def metrics(location_list, total_measurements,nan_values):
    
    metrics = pd.DataFrame({})
    locations = ['Audubon', 'Adams', 'Balboa', 'Browne', 
            'Grant', 'Jefferson', 'Lidgerwood', 'Regal',
            'Sheridan', 'Stevens']
    
    average = []
    median = []
    stdev = []
    number_measurements = []
    percentage = []
    
    for i, total, nan_values in zip(location_list, total_measurements, nan_values):
        
        #print(i.head())
        
        location = i
        avg = np.mean(location['PM2_5_corrected'])
        med = np.median(location['PM2_5_corrected'])
        stdv = np.std(location['PM2_5_corrected'])
        measurements = len(i.index)
        percent_total = (measurements/(total-nan_values))*100
        average.append(avg)
        median.append(med)
        stdev.append(stdv)
        number_measurements.append(measurements)
        percentage.append(percent_total)
    
    metrics['Location'] = locations
    metrics['average'] = average
    metrics['median'] = median
    metrics['stdev'] = stdev
    metrics['measurements'] = number_measurements
    metrics['percentage'] = percentage
    
    metrics = metrics.sort_values('average', ascending=False)
    
    return metrics