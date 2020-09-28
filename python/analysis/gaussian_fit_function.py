#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 13:49:19 2020

@author: matthew
"""


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def gaussian_fit(location):
    
    
    # the if statement discerns whether or not the input location is a Clarity node (has 'prediction_residuals') or the indoor unit (has 'shifted_residuals')
    if 'prediction_residuals' in location.columns:
        print(1)
        plot_title = (location.iloc[0]['Location']) + ' ' + 'Residuals'
        residuals = location['prediction_residuals']
        
    else:
        pass

    
    std_dev = residuals.std()
    two_std_dev = 2*std_dev
    
    mean,std=norm.fit(residuals)
    
    figure = plt.figure()
    figure.suptitle(plot_title, fontsize = 16)
    plt.xlabel('Residual Value (ug/m3)', fontsize=14)
    plt.ylabel('Probability', fontsize=14)
    plt.hist(residuals, bins=30, density=True)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    plt.plot(x, y)
    plt.show()
    
    print(plot_title + ' one sigma ' + '= ' + str(std_dev))
    print(plot_title + ' two sigmas ' + '= ' + str(two_std_dev))
    
    