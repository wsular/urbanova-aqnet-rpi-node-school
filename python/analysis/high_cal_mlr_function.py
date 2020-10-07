#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:59:33 2020

@author: matthew
"""

import statsmodels.api as sm



def mlr_function_high_cal(model, location):
    
    
    X = location[['PM2_5','Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)  Rel_humid
    X = X.dropna()
    X = sm.add_constant(X)
    predictions = model.predict(X)
    location['PM2_5_corrected'] = predictions
    # Print out the statistics
    #print_model = model.summary()
    #print(print_model)


    return predictions
