#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 08:43:33 2020

@author: matthew
"""

# input Clarity and BAM dataframes and combine into one dataframe so can cut based on the blank measurement threshold
# then calc slope using same code as in Augusta comparison script
# then use the sigma_blank from Mark Rowe and the slope to calculate LOD

# LOD = 3 sigmablank/k

# sigma_blank = standard deviation of reference at blank conditions ( blank conditions = < 1 ug/m3 according to Sayahi paper or use the LOD from Mark Rowe at SRCAA for their BAM at )
# k = the slope of the linear relationship for each PMS sensor versus FEM concentrations

import numpy as np
from scipy import stats
import scipy
import copy                                         

def lod(clarity, bam, threshold):
    
    
    df = copy.deepcopy(clarity)
    df['bam'] = bam['PM2_5']
    
    
    # Calculate slope of clarity node vs reference BAM

    #the data
    x=np.array(df.bam)
    y=np.array(df.PM2_5) 
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y) 
    print('slope = ' , slope)
    r_squared1 = r_value**2
    print('r^2 = ' , r_squared1)

    # determine best fit line
    par = np.polyfit(x, y, 1, full=True)
    slope1=par[0][0]
    
    print('slope1 = ' , slope1)
    
   # intercept1=par[0][1]
   # y1_predicted = [slope1*i + intercept1  for i in x]
    
    
    df = df[df['bam'] < threshold]
    print('Number of Measurements = ' , len(df.index))
    sigma_blank = np.std(df['PM2_5'])
    print('sigma_blank = ' , sigma_blank)
    
    lod = (3*sigma_blank)/slope
    print('Limit of Detection = ' , lod)
    
    return lod