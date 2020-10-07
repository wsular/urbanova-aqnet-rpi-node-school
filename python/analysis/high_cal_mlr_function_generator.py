#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:59:33 2020

@author: matthew
"""

import statsmodels.api as sm
import numpy as np
import scipy
from bokeh.plotting import figure
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
import pandas as pd
from glob import glob

def high_cal_setup():
    calibration_df = pd.DataFrame({})
    files = glob('/Users/matthew/Desktop/data/high_calibration/high_calibration*.csv')
    files.sort()
    for file in files:
        calibration_df = pd.concat([calibration_df, pd.read_csv(file)], sort=False)


    calibration_df['time'] = pd.to_datetime(calibration_df['time'])
    calibration_df = calibration_df.sort_values('time')
    calibration_df.index = calibration_df.time
    calibration_df = calibration_df.dropna()


    
    return(calibration_df)


def generate_mlr_function_high_cal(high_cal_df, location_name):
    
    if location_name == 'Audubon':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Audubon', 'Audubon_rh', 'Audubon_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)


    if location_name == 'Adams':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Adams', 'Adams_rh', 'Adams_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)
        
    if location_name == 'Balboa':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Balboa', 'Balboa_rh', 'Balboa_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)
        

    if location_name == 'Browne':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Browne', 'Browne_rh', 'Browne_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)


    if location_name == 'Grant':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Grant', 'Grant_rh', 'Grant_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)

    
    if location_name == 'Jefferson':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Jefferson', 'Jefferson_rh', 'Jefferson_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)


    if location_name == 'Lidgerwood':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Lidgerwood', 'Lidgerwood_rh', 'Lidgerwood_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)


    if location_name == 'Regal':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Regal', 'Regal_rh', 'Regal_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)

    
    if location_name == 'Sheridan':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Sheridan', 'Sheridan_rh', 'Sheridan_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)


    if location_name == 'Stevens':
    
        y = high_cal_df['ref_avg']
        X = high_cal_df[['Stevens', 'Stevens_rh', 'Stevens_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
        model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
        predictions = model.predict(X)

        # Print out the statistics
        print_model = model.summary()
        print(print_model)
        
    #print(type(predictions))   

    # Mean Error Calc and performance stats
    print('\n')
    print('Ref_Avg average = ', np.mean(high_cal_df['ref_avg']), '\n')
    print('Ref_Avg median = ', np.median(high_cal_df['ref_avg']), '\n')
    print('Ref_Avg sum = ', np.sum(high_cal_df['ref_avg']), '\n')
   
    print(location_name + ' Adj average = ', predictions.mean(), '\n')
    print(location_name + ' Adj median = ', predictions.median(), '\n')
    print(location_name + ' Adj sum = ', predictions.sum(), '\n')
    
    print(location_name +  ' Adj RMSE = ', np.sqrt((np.sum((predictions-high_cal_df['ref_avg'])**2)/len(predictions))), '\n')
    
    mae = (abs(high_cal_df['ref_avg']-predictions).sum())/(high_cal_df['ref_avg'].count())
        
    print(location_name + ' mean absolute error =', mae, '\n')
        
    
    #return predictions     used when adding mlr predictions to calibration df in high_calibration script
    return model
