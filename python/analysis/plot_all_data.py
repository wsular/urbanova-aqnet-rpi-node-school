#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:36:49 2020

@author: matthew
"""

#%%
import pandas as pd
from glob import glob
import numpy as np
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from scipy import stats
from bokeh.io import export_png, output_file
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure
from bokeh.models import Range1d, LinearAxis
import importlib
from bokeh.layouts import column
from bokeh.plotting import reset_output

import holoviews as hv
hv.extension('bokeh', logo=False)
import geoviews as gv
import geoviews.tile_sources as gvts
from bokeh.models import LogColorMapper, LogTicker, ColorBar
from bokeh.layouts import row
from bokeh.models import Label, LabelSet
from datetime import datetime, timedelta
from itertools import combinations

#import import_test
#from import_test import y

import copy

from random_forest_function_test import rf, evaluate_model
from mlr_function import mlr_function, mlr_model
from hybrid_function import hybrid_function
from rf_uncertainty_function import rf_uncertainty
from mlr_uncertainty_function import mlr_uncertainty
from daily_random_forest_function import daily_random_forest, daily_rf
from daily_mlr_function import daily_mlr_function, daily_mlr_model
from daily_rf_uncertainty_function import daily_rf_uncertainty
from daily_mlr_uncertainty_function import daily_mlr_uncertainty
from Augusta_BAM_uncertainty import Augusta_BAM_uncertainty
from Augusta_mlr_indoor_uncertainty import indoor_mlr_uncertainty
from plot_indoor_out_comparison import indoor_outdoor_plot
from indoor_outdoor_correlation import in_out_corr
from indoor_optimum_shift import opt_shift
from good_aqi import good_aqi
from moderate_aqi import moderate_aqi
from unhealthy_for_sensitive_aqi import unhealthy_for_sensitive_aqi
from unhealthy_aqi import unhealthy_aqi
from aqi_metrics import metrics
from spec_humid import spec_humid
from load_indoor_data import load_indoor
from uncertainty_compare import uncertainty_compare
from plot_all_function import plot_all
from uncertainty_compare_plot_function import plot_stat_diff
from gaussian_fit_function import gaussian_fit
from indoor_shift_outdoor_residuals import in_out_histogram
from high_cal_mlr_function_generator import high_cal_setup, generate_mlr_function_high_cal
from high_cal_mlr_function import mlr_function_high_cal
#%%

# initiate dataframe for high calibration data used to generate high calibration mlr functions for each location
calibration_df = high_cal_setup()

# generate the mlr for each location based on high calibration data
mlr_high_audubon = generate_mlr_function_high_cal(calibration_df, 'Audubon')
mlr_high_adams = generate_mlr_function_high_cal(calibration_df, 'Adams')
mlr_high_balboa = generate_mlr_function_high_cal(calibration_df, 'Balboa')
mlr_high_browne = generate_mlr_function_high_cal(calibration_df, 'Browne')
mlr_high_grant = generate_mlr_function_high_cal(calibration_df, 'Grant')
mlr_high_jefferson = generate_mlr_function_high_cal(calibration_df, 'Jefferson')
mlr_high_lidgerwood = generate_mlr_function_high_cal(calibration_df, 'Lidgerwood')
mlr_high_regal = generate_mlr_function_high_cal(calibration_df, 'Regal')
mlr_high_sheridan = generate_mlr_function_high_cal(calibration_df, 'Sheridan')
mlr_high_stevens = generate_mlr_function_high_cal(calibration_df, 'Stevens')
#%%
PlotType = 'HTMLfile'

ModelType = 'mlr'    # options: rf, mlr, hybrid, linear
stdev_number = 1   # defines whether using 1 or 2 stdev for uncertainty

slope_sigma1 = 2       # percent uncertainty of SRCAA BAM calibration to reference clarity slope
slope_sigma2 = 4.5     # percent uncertainty of slope for paccar roof calibrations (square root of VAR slope from excel)
slope_sigma_paccar = 2     # percent uncertainty of slope for Paccar Clarity unit at SRCAA BAM calibration
sigma_i = 5            # uncertainty of Clarity measurements (arbitrary right now) in ug/m^3


# Choose dates of interest
#Augusta Times
#start_time = '2019-12-17 00:00'
#end_time = '2020-03-05 00:00'

# For indoor unit resampling
#start_time = '2020-02-16 07:00'
#end_time = '2020-06-22 07:00'

# Stage 1 burn ban
#start_time = '2019-10-31 07:00'
#end_time = '2019-11-08 19:00'

# Complete sampling time
#start_time = '2020-03-10 07:00'
#end_time = '2020-09-10 10:00'

# Date Range of interest
start_time = '2020-09-10 07:00'   # was 2/9//20 7:00
end_time = '2020-09-21 07:00'

#interval = '2T'    # for plotting indoor/outdoor comparisons
interval = '60T'
#interval = '15T'  # only used for resampling indoor data so more manageable and doesnt take 20 min to load in...only use 1 hr and 24 hr for any analysis as these are what the calibrations are based on 
#interval = '24H'

### DON'T USE RESAMPLE FOR STATS TABLE (WHEN USING 60T RESAMPLE, DIDN'T CALCULATE MEDIAN OR VARIANCE)

### USE INTERVAL IF LOOKING FOR MOVING AVERAGE FOR 24 HR EXCEEDANCE CALCULATED AT BOTTOM OF SCRIPT


#%%

# Import location traits

location_traits = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/location_traits.csv')
files.sort()
for file in files:
    location_traits = pd.concat([location_traits, pd.read_csv(file)], sort=False)


#Import entire data set

Audubon_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Audubon*.csv')
files.sort()
for file in files:
    Audubon_All = pd.concat([Audubon_All, pd.read_csv(file)], sort=False)
#Audubon_All['PM2_5_corrected'] = (Audubon_All['PM2_5']-0.42073)/1.0739

#Audubon_All['PM2_5_corrected'] = np.where((Audubon_All.PM2_5 > 50), (Audubon_All.PM2_5-36.06)/0.55, Audubon_All.PM2_5)  # high calibration adjustment
Audubon_All['PM2_5_corrected'] = np.where((Audubon_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_audubon, Audubon_All), Audubon_All.PM2_5)  # high calibration adjustment
Audubon_All['PM2_5_corrected'] = np.where((Audubon_All.PM2_5 < 100), (Audubon_All.PM2_5-0.4207)/1.0739, Audubon_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Audubon_All['PM2_5_corrected'] = (Audubon_All['PM2_5_corrected']+0.5693)/1.9712    # adjustment to Augusta BAM 
else:
    pass


Adams_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Adams*.csv')
files.sort()
for file in files:
    Adams_All = pd.concat([Adams_All, pd.read_csv(file)], sort=False)
#Adams_All['PM2_5_corrected'] = (Adams_All['PM2_5']+0.93)/1.1554
#Adams_All['PM2_5_corrected'] = np.where((Adams_All.PM2_5 > 50), (Adams_All.PM2_5-36.45)/1.03, Adams_All.PM2_5)  # high calibration adjustment

Adams_All['PM2_5_corrected'] = np.where((Adams_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_adams, Adams_All), Adams_All.PM2_5)  # high calibration adjustment
Adams_All['PM2_5_corrected'] = np.where((Adams_All.PM2_5 < 100), (Adams_All.PM2_5+0.93)/1.1554, Adams_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Adams_All['PM2_5_corrected'] = (Adams_All['PM2_5_corrected']+0.5693)/1.9712  # From AUGUSTA BAM comparison
else:
    pass


Balboa_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Balboa*.csv')
files.sort()
for file in files:
    Balboa_All = pd.concat([Balboa_All, pd.read_csv(file)], sort=False)
#Balboa_All['PM2_5_corrected'] = (Balboa_All['PM2_5']-0.2878)/1.2457       # works, checked with calculator
#Balboa_All['PM2_5_corrected'] = np.where((Balboa_All.PM2_5 > 50), (Balboa_All.PM2_5-20.74)/1.41, Balboa_All.PM2_5)  # high calibration adjustment

Balboa_All['PM2_5_corrected'] = np.where((Balboa_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_balboa, Balboa_All), Balboa_All.PM2_5)  # high calibration adjustment
Balboa_All['PM2_5_corrected'] = np.where((Balboa_All.PM2_5 < 100), (Balboa_All.PM2_5-0.2878)/1.2457, Balboa_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Balboa_All['PM2_5_corrected'] = (Balboa_All['PM2_5_corrected']+0.5693)/1.9712
else:
    pass


Browne_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Browne*.csv')
files.sort()
for file in files:
    Browne_All = pd.concat([Browne_All, pd.read_csv(file)], sort=False)
#Browne_All['PM2_5_corrected'] = (Browne_All['PM2_5']-0.4771)/1.1082       # works, checked with calculator
#Browne_All['PM2_5_corrected'] = np.where((Browne_All.PM2_5 > 50), (Browne_All.PM2_5-42.09)/0.85, Browne_All.PM2_5)  # high calibration adjustment

Browne_All['PM2_5_corrected'] = np.where((Browne_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_browne, Browne_All), Browne_All.PM2_5)  # high calibration adjustment
Browne_All['PM2_5_corrected'] = np.where((Browne_All.PM2_5 < 100), (Browne_All.PM2_5-0.4771)/1.1082, Browne_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Browne_All['PM2_5_corrected'] = (Browne_All['PM2_5_corrected']+0.5693)/1.9712
else:
    pass

Browne_All = Browne_All[Browne_All['PM2_5_corrected'] < 300]


Grant_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Grant*.csv')
files.sort()
for file in files:
    Grant_All = pd.concat([Grant_All, pd.read_csv(file)], sort=False)
#Grant_All['PM2_5_corrected'] = (Grant_All['PM2_5']+1.0965)/1.29
#Grant_All['PM2_5_corrected'] = np.where((Grant_All.PM2_5 > 50), (Grant_All.PM2_5-37.25)/1.16, Grant_All.PM2_5)  # high calibration adjustment

Grant_All['PM2_5_corrected'] = np.where((Grant_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_grant, Grant_All), Grant_All.PM2_5)  # high calibration adjustment
Grant_All['PM2_5_corrected'] = np.where((Grant_All.PM2_5 < 100), (Grant_All.PM2_5+1.0965)/1.29, Grant_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Grant_All['PM2_5_corrected'] = (Grant_All['PM2_5_corrected']+0.5693)/1.9712   # From AUGUSTA BAM comparison
else:
    pass


Jefferson_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Jefferson*.csv')
files.sort()
for file in files:
    Jefferson_All = pd.concat([Jefferson_All, pd.read_csv(file)], sort=False)
#Jefferson_All['PM2_5_corrected'] = (Jefferson_All['PM2_5']+0.7099)/1.1458
#Jefferson_All['PM2_5_corrected'] = np.where((Jefferson_All.PM2_5 > 50), (Jefferson_All.PM2_5-37.85)/0.96, Jefferson_All.PM2_5)  # high calibration adjustment

Jefferson_All['PM2_5_corrected'] = np.where((Jefferson_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_jefferson, Jefferson_All), Jefferson_All.PM2_5)  # high calibration adjustment
Jefferson_All['PM2_5_corrected'] = np.where((Jefferson_All.PM2_5 < 100), (Jefferson_All.PM2_5+0.7099)/1.1458, Jefferson_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Jefferson_All['PM2_5_corrected'] = (Jefferson_All['PM2_5_corrected']+0.5693)/1.9712   # From AUGUSTA BAM comparison
else:
    pass


Lidgerwood_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Lidgerwood*.csv')
files.sort()
for file in files:
    Lidgerwood_All = pd.concat([Lidgerwood_All, pd.read_csv(file)], sort=False)
#Lidgerwood_All['PM2_5_corrected'] = (Lidgerwood_All['PM2_5']-1.1306)/0.9566    # works, checked with calculator
#Lidgerwood_All['PM2_5_corrected'] = np.where((Lidgerwood_All.PM2_5 > 50), (Lidgerwood_All.PM2_5-29.33)/0.95, Lidgerwood_All.PM2_5)  # high calibration adjustment

Lidgerwood_All['PM2_5_corrected'] = np.where((Lidgerwood_All.PM2_5 > 100),  mlr_function_high_cal(mlr_high_lidgerwood, Lidgerwood_All), Lidgerwood_All.PM2_5)  # high calibration adjustment
Lidgerwood_All['PM2_5_corrected'] = np.where((Lidgerwood_All.PM2_5 < 100), (Lidgerwood_All.PM2_5-1.1306)/0.9566, Lidgerwood_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Lidgerwood_All['PM2_5_corrected'] = (Lidgerwood_All['PM2_5_corrected']+0.5693)/1.9712
else:
    pass


Regal_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Regal*.csv')
files.sort()
for file in files:
    Regal_All = pd.concat([Regal_All, pd.read_csv(file)], sort=False)
#Regal_All['PM2_5_corrected'] = (Regal_All['PM2_5']-0.247)/0.9915          # works, checked with calculator
#Regal_All['PM2_5_corrected'] = np.where((Regal_All.PM2_5 > 50), (Regal_All.PM2_5-29.86)/0.78, Regal_All.PM2_5)  # high calibration adjustment

Regal_All['PM2_5_corrected'] = np.where((Regal_All.PM2_5 > 100),  mlr_function_high_cal(mlr_high_regal, Regal_All), Regal_All.PM2_5)  # high calibration adjustment
Regal_All['PM2_5_corrected'] = np.where((Regal_All.PM2_5 < 100), (Regal_All.PM2_5-0.247)/0.9915, Regal_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Regal_All['PM2_5_corrected'] = (Regal_All['PM2_5_corrected']+0.5693)/1.9712
else:
    pass

    
Sheridan_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Sheridan*.csv')
files.sort()
for file in files:
    Sheridan_All = pd.concat([Sheridan_All, pd.read_csv(file)], sort=False)
#Sheridan_All['PM2_5_corrected'] = (Sheridan_All['PM2_5']+0.6958)/1.1468 
#Sheridan_All['PM2_5_corrected'] = np.where((Sheridan_All.PM2_5 > 50), (Sheridan_All.PM2_5-42.46)/0.97, Sheridan_All.PM2_5)  # high calibration adjustment

Sheridan_All['PM2_5_corrected'] = np.where((Sheridan_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_sheridan, Sheridan_All), Sheridan_All.PM2_5)  # high calibration adjustment
Sheridan_All['PM2_5_corrected'] = np.where((Sheridan_All.PM2_5 < 100), (Sheridan_All.PM2_5+0.6958)/1.1468, Sheridan_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Sheridan_All['PM2_5_corrected'] = (Sheridan_All['PM2_5_corrected']+0.5693)/1.9712   # From AUGUSTA BAM comparison
else:
    pass


Stevens_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Stevens*.csv')
files.sort()
for file in files:
    Stevens_All = pd.concat([Stevens_All, pd.read_csv(file)], sort=False)
#Stevens_All['PM2_5_corrected'] = (Stevens_All['PM2_5']+0.8901)/1.2767
#Stevens_All['PM2_5_corrected'] = np.where((Stevens_All.PM2_5 > 50), (Stevens_All.PM2_5-39.23)/1.2, Stevens_All.PM2_5)  # high calibration adjustment

Stevens_All['PM2_5_corrected'] = np.where((Stevens_All.PM2_5 > 100), mlr_function_high_cal(mlr_high_stevens, Stevens_All), Stevens_All.PM2_5)  # high calibration adjustment
Stevens_All['PM2_5_corrected'] = np.where((Stevens_All.PM2_5 < 100), (Stevens_All.PM2_5+0.8901)/1.2767, Stevens_All.PM2_5_corrected)  # Paccar roof adjustment


if ModelType == 'linear':
    Stevens_All['PM2_5_corrected'] = (Stevens_All['PM2_5_corrected']+0.5693)/1.9712   # From AUGUSTA BAM comparison
else:
    pass


Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv')
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)
Reference_All['PM2_5_corrected'] = (Reference_All['PM2_5'] + 0.6232)/1.7588   # From AUGUSTA BAM comparison
Reference_All['lower_uncertainty'] = Reference_All['PM2_5_corrected']-(Reference_All['PM2_5_corrected']*(((((((sigma_i/Reference_All['PM2_5_corrected'])*100))**2+slope_sigma1**2))**0.5)/100))
Reference_All['upper_uncertainty'] = Reference_All['PM2_5_corrected']+(Reference_All['PM2_5_corrected']*(((((((sigma_i/Reference_All['PM2_5_corrected'])*100))**2+slope_sigma1**2))**0.5)/100))


Paccar_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Paccar*.csv')
files.sort()
for file in files:
    Paccar_All = pd.concat([Paccar_All, pd.read_csv(file)], sort=False)
Paccar_All['PM2_5_corrected'] = (Paccar_All['PM2_5'] + 0.8256)/1.9127    # Correction from AUGUSTA BAM, works, checked with calculator
Paccar_All['lower_uncertainty'] = Paccar_All['PM2_5_corrected']-(Paccar_All['PM2_5_corrected']*(((((((sigma_i/Paccar_All['PM2_5_corrected'])*100))**2+slope_sigma_paccar**2))**0.5)/100))
Paccar_All['upper_uncertainty'] = Paccar_All['PM2_5_corrected']+(Paccar_All['PM2_5_corrected']*(((((((sigma_i/Paccar_All['PM2_5_corrected'])*100))**2+slope_sigma_paccar**2))**0.5)/100))


#Read in SRCAA Augusta site BAM data

Augusta_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)
Augusta_All['PM2_5_corrected'] = Augusta_All['PM2_5']    # creates column with same values so loops work below
Augusta_All['Location'] = 'Augusta'

Augusta_BAM_uncertainty(stdev_number,Augusta_All)

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time] 


Monroe_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Monroe_Neph/Monroe*.csv')
files.sort()
for file in files:
    Monroe_All = pd.concat([Monroe_All, pd.read_csv(file)], sort=False)
Monroe_All['PM2_5_corrected'] = Monroe_All['PM2_5']    # creates column with same values so loops work below in stats section
 
Broadway_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Broadway_BAM/Broadway*.csv')
files.sort()
for file in files:
    Broadway_All = pd.concat([Broadway_All, pd.read_csv(file)], sort=False)
Broadway_All['PM2_5_corrected'] = Broadway_All['PM2_5']    # creates column with same values so loops work below in stats section

Greenbluff_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Greenbluff_TEOM/Greenbluff*.csv')
files.sort()
for file in files:
    Greenbluff_All = pd.concat([Greenbluff_All, pd.read_csv(file)], sort=False)
Greenbluff_All['PM2_5_corrected'] = Greenbluff_All['PM2_5']    # creates column with same values so loops work below in stats section


#%%

# winter mlr calibration for low concentrations

# using typed out equation rather than mlr function because function was returning entire column rather than just the 
# calibration performed on measurements less than 100 (ie was affecting the high measurements as well)
# note that all the constants are the same because the same mlr equation derived from the Clarity reference node at 
# the Augusta site is being used for this calibration adjustment

Audubon_All['PM2_5_corrected1'] = np.where((Audubon_All.PM2_5 < 100), Audubon_All.PM2_5*0.454-Audubon_All.Rel_humid*0.0483-Audubon_All.temp*0.0774+4.8242, Audubon_All.PM2_5_corrected)  # high calibration adjustment
Adams_All['PM2_5_corrected'] = np.where((Adams_All.PM2_5 < 100), Adams_All.PM2_5*0.454-Adams_All.Rel_humid*0.0483-Adams_All.temp*0.0774+4.8242, Adams_All.PM2_5_corrected)  # high calibration adjustment
Balboa_All['PM2_5_corrected'] = np.where((Balboa_All.PM2_5 < 100), Balboa_All.PM2_5*0.454-Balboa_All.Rel_humid*0.0483-Balboa_All.temp*0.0774+4.8242, Balboa_All.PM2_5_corrected)  # high calibration adjustment
Browne_All['PM2_5_corrected'] = np.where((Browne_All.PM2_5 < 100), Browne_All.PM2_5*0.454-Browne_All.Rel_humid*0.0483-Browne_All.temp*0.0774+4.8242, Browne_All.PM2_5_corrected)  # high calibration adjustment
Grant_All['PM2_5_corrected'] = np.where((Grant_All.PM2_5 < 100), Grant_All.PM2_5*0.454-Grant_All.Rel_humid*0.0483-Grant_All.temp*0.0774+4.8242, Grant_All.PM2_5_corrected)  # high calibration adjustment
Jefferson_All['PM2_5_corrected'] = np.where((Jefferson_All.PM2_5 < 100), Jefferson_All.PM2_5*0.454-Jefferson_All.Rel_humid*0.0483-Jefferson_All.temp*0.0774+4.8242, Jefferson_All.PM2_5_corrected)  # high calibration adjustment
Lidgerwood_All['PM2_5_corrected'] = np.where((Lidgerwood_All.PM2_5 < 100), Lidgerwood_All.PM2_5*0.454-Lidgerwood_All.Rel_humid*0.0483-Lidgerwood_All.temp*0.0774+4.8242, Lidgerwood_All.PM2_5_corrected)  # high calibration adjustment
Regal_All['PM2_5_corrected'] = np.where((Regal_All.PM2_5 < 100), Regal_All.PM2_5*0.454-Regal_All.Rel_humid*0.0483-Regal_All.temp*0.0774+4.8242, Regal_All.PM2_5_corrected)  # high calibration adjustment
Sheridan_All['PM2_5_corrected'] = np.where((Sheridan_All.PM2_5 < 100), Sheridan_All.PM2_5*0.454-Sheridan_All.Rel_humid*0.0483-Sheridan_All.temp*0.0774+4.8242, Sheridan_All.PM2_5_corrected)  # high calibration adjustment
Stevens_All['PM2_5_corrected'] = np.where((Stevens_All.PM2_5 < 100), Stevens_All.PM2_5*0.454-Stevens_All.Rel_humid*0.0483-Stevens_All.temp*0.0774+4.8242, Stevens_All.PM2_5_corrected)  # high calibration adjustment


# the commented lines below affect the high measurements as well so dont use
#Audubon_All['PM2_5_corrected1'] = np.where((Audubon_All.PM2_5 < 100), mlr_function(mlr_model, Audubon_All), Audubon_All.PM2_5_corrected)  # high calibration adjustment
#Adams_All['PM2_5_corrected'] = np.where((Adams_All.PM2_5 < 100), mlr_function(mlr_model, Adams_All), Adams_All.PM2_5_corrected)  # high calibration adjustment
#Balboa_All['PM2_5_corrected'] = np.where((Balboa_All.PM2_5 < 100), mlr_function(mlr_model, Balboa_All), Balboa_All.PM2_5_corrected)  # high calibration adjustment
#Browne_All['PM2_5_corrected'] = np.where((Browne_All.PM2_5 < 100), mlr_function(mlr_model, Browne_All), Browne_All.PM2_5_corrected)  # high calibration adjustment
#Grant_All['PM2_5_corrected'] = np.where((Grant_All.PM2_5 < 100), mlr_function(mlr_model, Grant_All), Grant_All.PM2_5_corrected)  # high calibration adjustment
#Jefferson_All['PM2_5_corrected'] = np.where((Jefferson_All.PM2_5 < 100), mlr_function(mlr_model, Jefferson_All), Jefferson_All.PM2_5_corrected)  # high calibration adjustment
#Lidgerwood_All['PM2_5_corrected'] = np.where((Lidgerwood_All.PM2_5 < 100), mlr_function(mlr_model, Lidgerwood_All), Lidgerwood_All.PM2_5_corrected)  # high calibration adjustment
#Regal_All['PM2_5_corrected'] = np.where((Regal_All.PM2_5 < 100), mlr_function(mlr_model, Regal_All), Regal_All.PM2_5_corrected)  # high calibration adjustment
#Sheridan_All['PM2_5_corrected'] = np.where((Sheridan_All.PM2_5 < 100), mlr_function(mlr_model, Sheridan_All), Sheridan_All.PM2_5_corrected)  # high calibration adjustment
#Stevens_All['PM2_5_corrected'] = np.where((Stevens_All.PM2_5 < 100), mlr_function(mlr_model, Stevens_All), Stevens_All.PM2_5_corrected)  # high calibration adjustment

# resample and cut data to time period of interest

Audubon_All['time'] = pd.to_datetime(Audubon_All['time'])
Audubon_All = Audubon_All.sort_values('time')
Audubon_All.index = Audubon_All.time
Audubon = Audubon_All.loc[start_time:end_time]


Adams_All['time'] = pd.to_datetime(Adams_All['time'])
Adams_All = Adams_All.sort_values('time')
Adams_All.index = Adams_All.time
Adams = Adams_All.loc[start_time:end_time]


Balboa_All['time'] = pd.to_datetime(Balboa_All['time'])
Balboa_All = Balboa_All.sort_values('time')
Balboa_All.index = Balboa_All.time
Balboa = Balboa_All.loc[start_time:end_time]  


Browne_All['time'] = pd.to_datetime(Browne_All['time'])
Browne_All = Browne_All.sort_values('time')
Browne_All.index = Browne_All.time
Browne = Browne_All.loc[start_time:end_time]


Grant_All['time'] = pd.to_datetime(Grant_All['time'])
Grant_All = Grant_All.sort_values('time')
Grant_All.index = Grant_All.time
Grant = Grant_All.loc[start_time:end_time]

Grant_count = Grant.groupby(Grant.index.date).count()

Jefferson_All['time'] = pd.to_datetime(Jefferson_All['time'])
Jefferson_All = Jefferson_All.sort_values('time')
Jefferson_All.index = Jefferson_All.time
Jefferson = Jefferson_All.loc[start_time:end_time]


Lidgerwood_All['time'] = pd.to_datetime(Lidgerwood_All['time'])
Lidgerwood_All = Lidgerwood_All.sort_values('time')
Lidgerwood_All.index = Lidgerwood_All.time
Lidgerwood = Lidgerwood_All.loc[start_time:end_time]


Regal_All['time'] = pd.to_datetime(Regal_All['time'])
Regal_All = Regal_All.sort_values('time')
Regal_All.index = Regal_All.time
Regal = Regal_All.loc[start_time:end_time]


Sheridan_All['time'] = pd.to_datetime(Sheridan_All['time'])
Sheridan_All = Sheridan_All.sort_values('time')
Sheridan_All.index = Sheridan_All.time
Sheridan = Sheridan_All.loc[start_time:end_time]


Stevens_All['time'] = pd.to_datetime(Stevens_All['time'])
Stevens_All = Stevens_All.sort_values('time')
Stevens_All.index = Stevens_All.time
Stevens = Stevens_All.loc[start_time:end_time]


Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]


Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar = Paccar_All.loc[start_time:end_time]
Paccar_count = Paccar.resample('15T').apply({'PM2_5':'count'})

    
Monroe_All['time'] = pd.to_datetime(Monroe_All['time'])
Monroe_All = Monroe_All.sort_values('time')
Monroe_All.index = Monroe_All.time
Monroe = Monroe_All.loc[start_time:end_time] 

Broadway_All['time'] = pd.to_datetime(Broadway_All['time'])
Broadway_All = Broadway_All.sort_values('time')
Broadway_All.index = Broadway_All.time
Broadway = Broadway_All.loc[start_time:end_time] 

Greenbluff_All['time'] = pd.to_datetime(Greenbluff_All['time'])
Greenbluff_All = Greenbluff_All.sort_values('time')
Greenbluff_All.index = Greenbluff_All.time
Greenbluff = Greenbluff_All.loc[start_time:end_time]

####  Uncomment these for resampling   ############
# use pad for 15 min interval and mean for 60 min interval 

Audubon = Audubon.resample(interval).mean() 
Adams = Adams.resample(interval).mean()  
Balboa = Balboa.resample(interval).mean()
Browne = Browne.resample(interval).mean()
Grant = Grant.resample(interval).mean()
Jefferson = Jefferson.resample(interval).mean()
Lidgerwood = Lidgerwood.resample(interval).mean()
Regal = Regal.resample(interval).mean()
Sheridan = Sheridan.resample(interval).mean()
Stevens = Stevens.resample(interval).mean()
Reference = Reference.resample(interval).mean()
Paccar = Paccar.resample(interval).mean()
Augusta = Augusta.resample(interval).mean()

Audubon['Location'] = 'Audubon'
Adams['Location'] = 'Adams'
Balboa['Location'] = 'Balboa'
Browne['Location'] = 'Browne'
Grant['Location'] = 'Grant'
Jefferson['Location'] = 'Jefferson'
Lidgerwood['Location'] = 'Lidgerwood'
Regal['Location'] = 'Regal'
Sheridan['Location'] = 'Sheridan'
Stevens['Location'] = 'Stevens'

#unit_1 = Reference
#unit_2 = Paccar

# Read in Radiosonde Data
radiosonde = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/radiosondes/inv_height_m.csv')
files.sort()
for file in files:
    radiosonde = pd.concat([radiosonde, pd.read_csv(file)], sort=False)

print(radiosonde.dtypes)

radiosonde['date_obj'] =  pd.to_datetime(radiosonde['datetime'])#, format='Y-%m-%dT%H:%M%:%SZ')
print(radiosonde.dtypes)

radiosonde['iso_date'] = radiosonde['date_obj'].apply(lambda x: x.isoformat())
radiosonde.index = radiosonde.iso_date
radiosonde['adams_PM2_5_corr'] = Adams['PM2_5_corrected']
radiosonde['adams_temp'] = Adams['temp']

#%%
# Load in Indoor unit data for specific  humidity calc

audubon_bme = pd.DataFrame({})
audubon_bme_json = pd.DataFrame({})
audubon_bme, audubon_bme_json = load_indoor('Audubon', audubon_bme,audubon_bme_json, interval, start_time, end_time)

adams_bme = pd.DataFrame({})
adams_bme_json = pd.DataFrame({})
adams_bme, adams_bme_json = load_indoor('Adams', adams_bme,adams_bme_json, interval, start_time, end_time)

balboa_bme = pd.DataFrame({})
balboa_bme_json = pd.DataFrame({})
balboa_bme, balboa_bme_json = load_indoor('Balboa', balboa_bme,balboa_bme_json, interval, start_time, end_time)

browne_bme = pd.DataFrame({})
browne_bme_json = pd.DataFrame({})
browne_bme, browne_bme_json = load_indoor('Browne', browne_bme,browne_bme_json, interval, start_time, end_time)

grant_bme = pd.DataFrame({})
grant_bme_json = pd.DataFrame({})
grant_bme, grant_bme_json = load_indoor('Grant', grant_bme,grant_bme_json, interval, start_time, end_time)

jefferson_bme = pd.DataFrame({})
jefferson_bme_json = pd.DataFrame({})
jefferson_bme, jefferson_bme_json = load_indoor('Jefferson', jefferson_bme,jefferson_bme_json, interval, start_time, end_time)

lidgerwood_bme = pd.DataFrame({})
lidgerwood_bme_json = pd.DataFrame({})
lidgerwood_bme, lidgerwood_bme_json = load_indoor('Lidgerwood', lidgerwood_bme,lidgerwood_bme_json, interval, start_time, end_time)

regal_bme = pd.DataFrame({})
regal_bme_json = pd.DataFrame({})
regal_bme, regal_bme_json = load_indoor('Regal', regal_bme,regal_bme_json, interval, start_time, end_time)

sheridan_bme = pd.DataFrame({})
sheridan_bme_json = pd.DataFrame({})
sheridan_bme, sheridan_bme_json = load_indoor('Sheridan', sheridan_bme,sheridan_bme_json, interval, start_time, end_time)

stevens_bme = pd.DataFrame({})
stevens_bme_json = pd.DataFrame({})
stevens_bme, stevens_bme_json = load_indoor('Stevens', stevens_bme,stevens_bme_json, interval, start_time, end_time)

# Add specific humidity column to corresponding Clarity nodes
#%%
spec_humid(audubon_bme, audubon_bme_json, Audubon)
spec_humid(adams_bme, adams_bme_json, Adams)
spec_humid(balboa_bme, balboa_bme_json, Balboa)
spec_humid(browne_bme, browne_bme_json, Browne)
spec_humid(regal_bme, regal_bme_json, Grant)       # Note Grant using Regal BME to fill in gaps of pressure data when Grant BME down
spec_humid(adams_bme, adams_bme_json, Jefferson)    # Note Jefferson using Adams BME to fill in gaps of pressure data when Jefferson BME down
spec_humid(balboa_bme, balboa_bme_json, Lidgerwood) # Note Lidgerwood using Balboa BME to fill in gaps of pressure data when Lidgerwood BME down
spec_humid(regal_bme, regal_bme_json, Regal)
spec_humid(sheridan_bme, sheridan_bme_json, Sheridan)
spec_humid(stevens_bme, stevens_bme_json, Stevens)

#%%

# this was used for testing different calibration methods
# as of 10/5/20, using the mlr calibrations above so don't run this cell

# Use RF calibration 

if ModelType=='rf' and interval=='60T':
    evaluate_model(rf, Audubon)
    evaluate_model(rf, Adams)
    evaluate_model(rf, Balboa)
    evaluate_model(rf, Browne)
    evaluate_model(rf, Grant) 
    evaluate_model(rf, Jefferson)
    evaluate_model(rf, Lidgerwood)
    evaluate_model(rf, Regal)
    evaluate_model(rf, Sheridan)
    evaluate_model(rf, Stevens)
    
else:
    pass

if ModelType=='rf' and interval=='24H':
    daily_random_forest(daily_rf, Audubon)
    daily_random_forest(daily_rf, Adams)
    daily_random_forest(daily_rf, Balboa)
    daily_random_forest(daily_rf, Browne)
    daily_random_forest(daily_rf, Grant) 
    daily_random_forest(daily_rf, Jefferson)
    daily_random_forest(daily_rf, Lidgerwood)
    daily_random_forest(daily_rf, Regal)
    daily_random_forest(daily_rf, Sheridan)
    daily_random_forest(daily_rf, Stevens)
    
else:
    pass

# Use mlr calibration


if ModelType=='mlr' and interval=='60T':


    mlr_function(mlr_model, Audubon)
    mlr_function(mlr_model, Adams)
    mlr_function(mlr_model, Balboa)
    mlr_function(mlr_model, Browne)
    mlr_function(mlr_model, Grant) 
    mlr_function(mlr_model, Jefferson)
    mlr_function(mlr_model, Lidgerwood)
    mlr_function(mlr_model, Regal)
    mlr_function(mlr_model, Sheridan)
    mlr_function(mlr_model, Stevens)
else:
    pass

if ModelType=='mlr' and interval=='24H':
    daily_mlr_function(daily_mlr_model, Audubon)
    daily_mlr_function(daily_mlr_model, Adams)
    daily_mlr_function(daily_mlr_model, Balboa)
    daily_mlr_function(daily_mlr_model, Browne)
    daily_mlr_function(daily_mlr_model, Grant) 
    daily_mlr_function(daily_mlr_model, Jefferson)
    daily_mlr_function(daily_mlr_model, Lidgerwood)
    daily_mlr_function(daily_mlr_model, Regal)
    daily_mlr_function(daily_mlr_model, Sheridan)
    daily_mlr_function(daily_mlr_model, Stevens)
else:
    pass
# Use hybrid calibration 
    
if ModelType=='hybrid' and interval=='60T':
    Audubon_All = hybrid_function(rf, mlr_model, Audubon)
    Adams_All = hybrid_function(rf, mlr_model, Adams)
    Balboa_All = hybrid_function(rf, mlr_model, Balboa)
    Browne_All = hybrid_function(rf, mlr_model, Browne)
    Grant_All = hybrid_function(rf, mlr_model, Grant) 
    Jefferson_All = hybrid_function(rf, mlr_model, Jefferson)
    Lidgerwood_All = hybrid_function(rf, mlr_model, Lidgerwood)
    Regal_All = hybrid_function(rf ,mlr_model, Regal)
    Sheridan_All = hybrid_function(rf, mlr_model, Sheridan)
    Stevens_All =  hybrid_function(rf, mlr_model, Stevens)
else:
    pass
#%%

# Add uncertainty bounds
    
if ModelType=='rf' and interval=='60T':
    rf_uncertainty(stdev_number,Audubon)
    rf_uncertainty(stdev_number,Adams)
    rf_uncertainty(stdev_number,Balboa)
    rf_uncertainty(stdev_number,Browne)
    rf_uncertainty(stdev_number,Grant)
    rf_uncertainty(stdev_number,Jefferson)
    rf_uncertainty(stdev_number,Lidgerwood)
    rf_uncertainty(stdev_number,Regal)
    rf_uncertainty(stdev_number,Sheridan)
    rf_uncertainty(stdev_number,Stevens)
else:
    pass

if ModelType=='mlr' and interval=='60T':
    mlr_uncertainty(stdev_number,Audubon)
    mlr_uncertainty(stdev_number,Adams)
    mlr_uncertainty(stdev_number,Balboa)
    mlr_uncertainty(stdev_number,Browne)
    mlr_uncertainty(stdev_number,Grant)
    mlr_uncertainty(stdev_number,Jefferson)
    mlr_uncertainty(stdev_number,Lidgerwood)
    mlr_uncertainty(stdev_number,Regal)
    mlr_uncertainty(stdev_number,Sheridan)
    mlr_uncertainty(stdev_number,Stevens)
else:
    pass

if ModelType=='rf' and interval=='24H':
    daily_rf_uncertainty(stdev_number,Audubon)
    daily_rf_uncertainty(stdev_number,Adams)
    daily_rf_uncertainty(stdev_number,Balboa)
    daily_rf_uncertainty(stdev_number,Browne)
    daily_rf_uncertainty(stdev_number,Grant)
    daily_rf_uncertainty(stdev_number,Jefferson)
    daily_rf_uncertainty(stdev_number,Lidgerwood)
    daily_rf_uncertainty(stdev_number,Regal)
    daily_rf_uncertainty(stdev_number,Sheridan)
    daily_rf_uncertainty(stdev_number,Stevens)
else:
    pass

if ModelType=='mlr' and interval=='24H':
    daily_mlr_uncertainty(stdev_number,Audubon)
    daily_mlr_uncertainty(stdev_number,Adams)
    daily_mlr_uncertainty(stdev_number,Balboa)
    daily_mlr_uncertainty(stdev_number,Browne)
    daily_mlr_uncertainty(stdev_number,Grant)
    daily_mlr_uncertainty(stdev_number,Jefferson)
    daily_mlr_uncertainty(stdev_number,Lidgerwood)
    daily_mlr_uncertainty(stdev_number,Regal)
    daily_mlr_uncertainty(stdev_number,Sheridan)
    daily_mlr_uncertainty(stdev_number,Stevens)
else:
    pass
#%%
#Audubon_All['lower_uncertainty'] = Audubon_All['PM2_5_corrected']-(Audubon_All['PM2_5_corrected']*((((2*(((sigma_i/Audubon_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Audubon_All['upper_uncertainty'] = Audubon_All['PM2_5_corrected']+(Audubon_All['PM2_5_corrected']*((((2*(((sigma_i/Audubon_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Adams_All['lower_uncertainty'] = Adams_All['PM2_5_corrected']-(Adams_All['PM2_5_corrected']*((((2*(((sigma_i/Adams_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Adams_All['upper_uncertainty'] = Adams_All['PM2_5_corrected']+(Adams_All['PM2_5_corrected']*((((2*(((sigma_i/Adams_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Balboa_All['lower_uncertainty'] = Balboa_All['PM2_5_corrected']-(Balboa_All['PM2_5_corrected']*((((2*(((sigma_i/Balboa_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Balboa_All['upper_uncertainty'] = Balboa_All['PM2_5_corrected']+(Balboa_All['PM2_5_corrected']*((((2*(((sigma_i/Balboa_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Browne_All['lower_uncertainty'] = Browne_All['PM2_5_corrected']-(Browne_All['PM2_5_corrected']*((((2*(((sigma_i/Browne_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Browne_All['upper_uncertainty'] = Browne_All['PM2_5_corrected']+(Browne_All['PM2_5_corrected']*((((2*(((sigma_i/Browne_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Grant_All['lower_uncertainty'] = Grant_All['PM2_5_corrected']-(Grant_All['PM2_5_corrected']*((((2*(((sigma_i/Grant_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Grant_All['upper_uncertainty'] = Grant_All['PM2_5_corrected']+(Grant_All['PM2_5_corrected']*((((2*(((sigma_i/Grant_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Jefferson_All['lower_uncertainty'] = Jefferson_All['PM2_5_corrected']-(Jefferson_All['PM2_5_corrected']*((((2*(((sigma_i/Jefferson_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Jefferson_All['upper_uncertainty'] = Jefferson_All['PM2_5_corrected']+(Jefferson_All['PM2_5_corrected']*((((2*(((sigma_i/Jefferson_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Lidgerwood_All['lower_uncertainty'] = Lidgerwood_All['PM2_5_corrected']-(Lidgerwood_All['PM2_5_corrected']*((((2*(((sigma_i/Lidgerwood_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Lidgerwood_All['upper_uncertainty'] = Lidgerwood_All['PM2_5_corrected']+(Lidgerwood_All['PM2_5_corrected']*((((2*(((sigma_i/Lidgerwood_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Regal_All['lower_uncertainty'] = Regal_All['PM2_5_corrected']-(Regal_All['PM2_5_corrected']*((((2*(((sigma_i/Regal_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Regal_All['upper_uncertainty'] = Regal_All['PM2_5_corrected']+(Regal_All['PM2_5_corrected']*((((2*(((sigma_i/Regal_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Sheridan_All['lower_uncertainty'] = Sheridan_All['PM2_5_corrected']-(Sheridan_All['PM2_5_corrected']*((((2*(((sigma_i/Sheridan_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Sheridan_All['upper_uncertainty'] = Sheridan_All['PM2_5_corrected']+(Sheridan_All['PM2_5_corrected']*((((2*(((sigma_i/Sheridan_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))

#Stevens_All['lower_uncertainty'] = Stevens_All['PM2_5_corrected']-(Stevens_All['PM2_5_corrected']*((((2*(((sigma_i/Stevens_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))
#Stevens_All['upper_uncertainty'] = Stevens_All['PM2_5_corrected']+(Stevens_All['PM2_5_corrected']*((((2*(((sigma_i/Stevens_All['PM2_5_corrected'])*100))**2+slope_sigma1**2+slope_sigma2**2))**0.5)/100))



#%%

# create dictionary for each location that contains that location compared to the other 9 and filters out all measurements that have overlapping uncertainties

df_dictionary = {'Audubon':Audubon, 'Adams':Adams, 'Balboa':Balboa, 'Browne':Browne, 
           'Grant':Grant, 'Jefferson':Jefferson, 'Lidgerwood':Lidgerwood, 
           'Regal':Regal, 'Sheridan':Sheridan, 'Stevens':Stevens}
#%%
# Note that after applying this function, the df_dictionary entries will have extra columns as a results of the function
# and only the last loop (Stevens) data is recorded (except for Stevens which has Sheridan data in the far right columns)

Audubon_filtered, Audubon_filtered_stats = uncertainty_compare(Audubon,df_dictionary)
Adams_filtered, Adams_filtered_stats = uncertainty_compare(Adams,df_dictionary)
Balboa_filtered, Balboa_filtered_stats = uncertainty_compare(Balboa,df_dictionary)
Browne_filtered, Browne_filtered_stats = uncertainty_compare(Browne,df_dictionary)
Grant_filtered, Grant_filtered_stats = uncertainty_compare(Grant,df_dictionary)
Jefferson_filtered, Jefferson_filtered_stats = uncertainty_compare(Jefferson,df_dictionary)
Lidgerwood_filtered, Lidgerwood_filtered_stats = uncertainty_compare(Lidgerwood,df_dictionary)
Regal_filtered, Regal_filtered_stats = uncertainty_compare(Regal,df_dictionary)
Sheridan_filtered, Sheridan_filtered_stats = uncertainty_compare(Sheridan,df_dictionary)
Stevens_filtered, Stevens_filtered_stats = uncertainty_compare(Stevens,df_dictionary)

#%%

# divide data into aqi categories (note that break points are based on 24 hour averages, so be sure to consider that if using hourly or 24 hr intervals)

# Good AQI 
Audubon_good_aqi = good_aqi(Audubon)
Adams_good_aqi = good_aqi(Adams)
Balboa_good_aqi = good_aqi(Balboa)
Browne_good_aqi = good_aqi(Browne)
Grant_good_aqi = good_aqi(Grant)
Jefferson_good_aqi = good_aqi(Jefferson)
Lidgerwood_good_aqi = good_aqi(Lidgerwood)
Regal_good_aqi = good_aqi(Regal)
Sheridan_good_aqi = good_aqi(Sheridan)
Stevens_good_aqi = good_aqi(Stevens)

good_aqi_list = [Audubon_good_aqi, Adams_good_aqi, Balboa_good_aqi, Browne_good_aqi, Grant_good_aqi,
                 Jefferson_good_aqi, Lidgerwood_good_aqi, Regal_good_aqi, Sheridan_good_aqi, Stevens_good_aqi]

total_measurements = [len(Audubon), len(Adams), len(Balboa), len(Browne), len(Grant), len(Jefferson), len(Lidgerwood),
                      len(Regal), len(Sheridan), len(Stevens)]

nan_values = [sum(pd.isnull(Audubon['PM2_5_corrected'])), sum(pd.isnull(Adams['PM2_5_corrected'])), sum(pd.isnull(Balboa['PM2_5_corrected'])),
              sum(pd.isnull(Browne['PM2_5_corrected'])), sum(pd.isnull(Grant['PM2_5_corrected'])), sum(pd.isnull(Jefferson['PM2_5_corrected'])),
              sum(pd.isnull(Lidgerwood['PM2_5_corrected'])), sum(pd.isnull(Regal['PM2_5_corrected'])), sum(pd.isnull(Sheridan['PM2_5_corrected'])),
              sum(pd.isnull(Stevens['PM2_5_corrected']))]

good_aqi_metrics = metrics(good_aqi_list, total_measurements, nan_values)

# checks for metrics function
#print(np.mean(Stevens_good_aqi['PM2_5_corrected']))
#print(np.median(Stevens_good_aqi['PM2_5_corrected']))
#print(np.std(Stevens_good_aqi['PM2_5_corrected']))

#%%

# Moderate AQI
Audubon_moderate_aqi = moderate_aqi(Audubon)
Adams_moderate_aqi = moderate_aqi(Adams)
Balboa_moderate_aqi = moderate_aqi(Balboa)
Browne_moderate_aqi = moderate_aqi(Browne)
Grant_moderate_aqi = moderate_aqi(Grant)
Jefferson_moderate_aqi = moderate_aqi(Jefferson)
Lidgerwood_moderate_aqi = moderate_aqi(Lidgerwood)
Regal_moderate_aqi = moderate_aqi(Regal)
Sheridan_moderate_aqi = moderate_aqi(Sheridan)
Stevens_moderate_aqi = moderate_aqi(Stevens)

moderate_aqi_list = [Audubon_moderate_aqi, Adams_moderate_aqi, Balboa_moderate_aqi, Browne_moderate_aqi, Grant_moderate_aqi,
                 Jefferson_moderate_aqi, Lidgerwood_moderate_aqi, Regal_moderate_aqi, Sheridan_moderate_aqi, Stevens_moderate_aqi]

moderate_aqi_metrics = metrics(moderate_aqi_list, total_measurements, nan_values)
#%%

# Unhealthy for Sensitive Groups AQI
Audubon_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Audubon)
Adams_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Adams)
Balboa_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Balboa)
Browne_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Browne)
Grant_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Grant)
Jefferson_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Jefferson)
Lidgerwood_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Lidgerwood)
Regal_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Regal)
Sheridan_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Sheridan)
Stevens_unhealthy_for_sensitive_aqi = unhealthy_for_sensitive_aqi(Stevens)

sensitive_aqi_list = [Audubon_unhealthy_for_sensitive_aqi, Adams_unhealthy_for_sensitive_aqi, Balboa_unhealthy_for_sensitive_aqi,
                      Browne_unhealthy_for_sensitive_aqi, Grant_unhealthy_for_sensitive_aqi, Jefferson_unhealthy_for_sensitive_aqi, 
                      Lidgerwood_unhealthy_for_sensitive_aqi, Regal_unhealthy_for_sensitive_aqi, Sheridan_unhealthy_for_sensitive_aqi, 
                      Stevens_unhealthy_for_sensitive_aqi]

sensitive_aqi_metrics = metrics(sensitive_aqi_list, total_measurements, nan_values)
#%%

# Unhealthy AQI
Audubon_unhealthy_aqi = unhealthy_aqi(Audubon)
Adams_unhealthy_aqi = unhealthy_aqi(Adams)
Balboa_unhealthy_aqi = unhealthy_aqi(Balboa)
Browne_unhealthy_aqi = unhealthy_aqi(Browne)
Grant_unhealthy_aqi = unhealthy_aqi(Grant)
Jefferson_unhealthy_aqi = unhealthy_aqi(Jefferson)
Lidgerwood_unhealthy_aqi = unhealthy_aqi(Lidgerwood)
Regal_unhealthy_aqi = unhealthy_aqi(Regal)
Sheridan_unhealthy_aqi = unhealthy_aqi(Sheridan)
Stevens_unhealthy_aqi = unhealthy_aqi(Stevens)

unhealthy_aqi_list = [Audubon_unhealthy_aqi, Adams_unhealthy_aqi, Balboa_unhealthy_aqi, Browne_unhealthy_aqi, Grant_unhealthy_aqi,
                 Jefferson_unhealthy_aqi, Lidgerwood_unhealthy_aqi, Regal_unhealthy_aqi, Sheridan_unhealthy_aqi, Stevens_unhealthy_aqi]

unhealthy_aqi_metrics = metrics(unhealthy_aqi_list, total_measurements, nan_values)
#%%
# Read in Airport Data

airport = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Airport_snow_depth.csv')
files.sort()
for file in files:
    airport = pd.concat([airport, pd.read_csv(file)], sort=False)

print(airport.dtypes)
#print(Paccar_All.dtypes)
airport['date_obj'] =  pd.to_datetime(airport['DATE'])#, format='Y-%m-%dT%H:%M%:%SZ')
print(airport.dtypes)

airport['iso_date'] = airport['date_obj'].apply(lambda x: x.isoformat())
airport.index = airport.iso_date

airport['snow_depth'] = airport['SNWD']


#%%

# Plot values for school combinations that dont have overlapping uncertainty errors (use date range starting from 9/1/19 to present)


plot_stat_diff(Audubon_filtered, df_dictionary)
#%%
plot_stat_diff(Adams_filtered, df_dictionary)
plot_stat_diff(Balboa_filtered, df_dictionary)
plot_stat_diff(Browne_filtered, df_dictionary)
plot_stat_diff(Grant_filtered, df_dictionary)
plot_stat_diff(Jefferson_filtered, df_dictionary)
plot_stat_diff(Lidgerwood_filtered, df_dictionary)
plot_stat_diff(Regal_filtered, df_dictionary)
plot_stat_diff(Sheridan_filtered, df_dictionary)
plot_stat_diff(Stevens_filtered, df_dictionary)

#%%
# Plot All

# Go into plot_all function for toggles on which uncertainties to display

plot_all(Audubon, Adams, Balboa, Browne, Grant, Jefferson, Lidgerwood, Regal, Sheridan,
         Stevens, Reference, Paccar, Augusta, Broadway, Greenbluff, Monroe)

#%%

# Plot PM2.5 vs Temp, inv height and lapse rate

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')

p1.title.text = 'PM2.5 vs Atm Conditions'    
p1.y_range = Range1d(start=0, end=50)

p1.line(Adams.index,     Adams.PM2_5_corrected,  legend='Adams PM 2.5 corr',     color='red',   muted_color='red', muted_alpha=0.1 , line_width=2)
p1.line(Adams.index,     Adams.temp,  legend='temp',       color='blue',   muted_color='blue', muted_alpha=0.1,  line_width=2)


p1.extra_y_ranges['Inv Height'] = Range1d(start=0, end=15000)
p1.add_layout(LinearAxis(y_range_name='Inv Height', axis_label=' Inv Height (m)'), 'right')
p1.line(radiosonde.date_obj,     radiosonde.inv_height,  legend='inv height',   y_range_name = 'Inv Height',      color='black',  muted_color='black', muted_alpha=0.1 ,  line_width=2)


p1.extra_y_ranges['Lapse Rate'] = Range1d(start=-100, end=100)
p1.add_layout(LinearAxis(y_range_name='Lapse Rate', axis_label=' Lapse Rate (ºC/km)'), 'right')

p1.line(radiosonde.date_obj,     radiosonde.lapse_rate,  legend='lapse rate', y_range_name = 'Lapse Rate',      color='green', muted_color='green', muted_alpha=0.1,   line_width=2)
p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="PM 2.5 vs Atm Conditions Time Series")




p2 = figure(plot_width=900,
            plot_height=450,
       #     x_axis_type='datetime',
            x_axis_label='Lapse Rate (ºC/km)',
            y_axis_label='PM 2.5 (ug/m^3)')

p2.title.text = 'PM2.5 vs Lapse Rate'    


#p2.scatter(radiosonde.Adams_temp,     radiosonde.Adams_PM2_5,  legend='Adams PM 2.5 corr',     color='red',   muted_color='red', muted_alpha=0.1 , line_width=2)
#p2.scatter(radiosonde.inv_height,     radiosonde.Adams_PM2_5,  legend='inv height',       color='blue',   muted_color='blue', muted_alpha=0.1,  line_width=2)
p2.scatter(radiosonde.lapse_rate,     radiosonde.Adams_PM2_5,       color='black',   muted_color='black', muted_alpha=0.1,  line_width=2)


p2.legend.click_policy="hide"
tab2 = Panel(child=p2, title="PM 2.5 vs Lapse Rate")

p3 = figure(plot_width=900,
            plot_height=450,
       #     x_axis_type='datetime',
            x_axis_label='Temp (ºC)',
            y_axis_label='PM 2.5 (ug/m3)')

p3.title.text = 'PM2.5 vs Adams Temp'    


p3.scatter(radiosonde.Adams_temp,     radiosonde.Adams_PM2_5,      color='black',   muted_color='red', muted_alpha=0.1 , line_width=2)


p4 = figure(plot_width=900,
            plot_height=450,
       #     x_axis_type='datetime',
            x_axis_label='Inversion Height (m)',
            y_axis_label='PM 2.5 (ug/m3)')

p4.title.text = 'PM2.5 vs Inv Height'    


p4.scatter(radiosonde.inv_height,     radiosonde.Adams_PM2_5,      color='black',   muted_color='red', muted_alpha=0.1 , line_width=2)


p5 = gridplot([[p2,p3, p4]], plot_width = 500, plot_height = 300)

tab3 = Panel(child=p5, title="PM 2.5 vs Atm Conditions")

tabs = Tabs(tabs=[ tab1, tab3])

show(tabs)

#%%

# Plot all Clarity PM 2.5 using interactive legend (mute)

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/clarity_PM2_5_time_series_legend_mute.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')

p1.title.text = 'Clarity Calibrated PM 2.5'

p1.line(Audubon.index,     Audubon.PM2_5_corrected,     legend='Audubon',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
p1.line(Adams.index,       Adams.PM2_5_corrected,       legend='Adams',         color='blue',        line_width=2, muted_color='blue', muted_alpha=0.2)
p1.line(Balboa.index,      Balboa.PM2_5_corrected,      legend='Balboa',        color='red',         line_width=2, muted_color='red', muted_alpha=0.2)
p1.line(Browne.index,      Browne.PM2_5_corrected,      legend='Browne',        color='black',       line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(Grant.index,       Grant.PM2_5_corrected,       legend='Grant',         color='purple',      line_width=2, muted_color='purple', muted_alpha=0.2)
p1.line(Jefferson.index,   Jefferson.PM2_5_corrected,   legend='Jefferson',     color='brown',       line_width=2, muted_color='brown', muted_alpha=0.2)
p1.line(Lidgerwood.index,  Lidgerwood.PM2_5_corrected,  legend='Lidgerwood',    color='orange',      line_width=2, muted_color='orange', muted_alpha=0.2)
p1.line(Regal.index,       Regal.PM2_5_corrected,       legend='Regal',         color='khaki',       line_width=2, muted_color='khaki', muted_alpha=0.2)
p1.line(Sheridan.index,    Sheridan.PM2_5_corrected,    legend='Sheridan',      color='deepskyblue', line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
p1.line(Stevens.index,     Stevens.PM2_5_corrected,     legend='Stevens',       color='grey',        line_width=2, muted_color='grey', muted_alpha=0.2)
p1.line(Reference.index,  Reference.PM2_5_corrected,    legend='Reference',     color='olive',       line_width=2, muted_color='olive', muted_alpha=0.2)
p1.line(Paccar.index,     Paccar.PM2_5_corrected,       legend='Paccar',        color='lime',        line_width=2, muted_color='lime', muted_alpha=0.2)
p1.line(Augusta.index,     Augusta.PM2_5,               legend='Augusta',       color='teal',         muted_color='teal', muted_alpha=0.2,     line_width=2)
p1.line(Broadway.index,    Broadway.PM2_5,              legend='Broadway BAM',  color='black',        muted_color='black', muted_alpha=0.2, line_width=2)
p1.line(Greenbluff.index,    Greenbluff.PM2_5,          legend='Greenbluff TEOM',  color='red',       muted_color='red', muted_alpha=0.2, line_width=2)
p1.line(Monroe.index,       Monroe.PM2_5,               legend='Monroe Neph',   color='blue',         muted_color='blue', muted_alpha=0.2, line_width=2)

p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Calibrated PM 2.5")

tabs = Tabs(tabs=[ tab1])

show(tabs)


#%%

# Plot all Clarity Temp using interactive legend (mute)

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/clarity_Temp_time_series_legend_mute.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='Temperature (ºC)')

p1.title.text = 'Clarity Temperature'

p1.line(Audubon.index,     Audubon.temp,     legend='Site #1',        color='green',             line_width=2, muted_color='green', muted_alpha=0.2)
p1.line(Adams.index,       Adams.temp,       legend='Site #2',        color='blue',              line_width=2, muted_color='blue', muted_alpha=0.2)
p1.line(Balboa.index,      Balboa.temp,      legend='Site #3',        color='red',               line_width=2, muted_color='red', muted_alpha=0.2)
p1.line(Browne.index,      Browne.temp,      legend='Site #4',        color='black',             line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(Grant.index,       Grant.temp,       legend='Site #5',        color='purple',            line_width=2, muted_color='purple', muted_alpha=0.2)
p1.line(Jefferson.index,   Jefferson.temp,   legend='Site #6',        color='brown',             line_width=2, muted_color='brown', muted_alpha=0.2)
p1.line(Lidgerwood.index,  Lidgerwood.temp,  legend='Site #7',        color='orange',            line_width=2, muted_color='orange', muted_alpha=0.2)
p1.line(Regal.index,       Regal.temp,       legend='Site #8',        color='khaki',             line_width=2, muted_color='khaki', muted_alpha=0.2)
p1.line(Sheridan.index,    Sheridan.temp,    legend='Site #9',        color='deepskyblue',       line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
p1.line(Stevens.index,     Stevens.temp,     legend='Site #10',       color='grey',              line_width=2, muted_color='grey', muted_alpha=0.2)
#p1.line(Reference.index,  Reference.temp,   legend='Reference',      color='olive',             line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Paccar.index,     Paccar.temp,      legend='Paccar',         color='lime',              line_width=2, muted_color='lime', muted_alpha=0.2)


p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Temperature")

tabs = Tabs(tabs=[ tab1])

show(tabs)

#%%

# Plot all Clarity RH using interactive legend (hide)

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/clarity_RH_time_series_legend_hide.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')

p1.title.text = 'Clarity Relative Humidity'        

p1.line(Audubon.index,     Audubon.Rel_humid,     legend='Site #1',       color='green',            line_width=2)
p1.line(Adams.index,       Adams.Rel_humid,       legend='Site #2',         color='blue',             line_width=2)
p1.line(Balboa.index,      Balboa.Rel_humid,      legend='Site #3',        color='red',              line_width=2)
p1.line(Browne.index,      Browne.Rel_humid,      legend='Site #4',        color='black',            line_width=2)
p1.line(Grant.index,       Grant.Rel_humid,       legend='Site #5',         color='purple',           line_width=2)
p1.line(Jefferson.index,   Jefferson.Rel_humid,   legend='Site #6',     color='brown',            line_width=2)
p1.line(Lidgerwood.index,  Lidgerwood.Rel_humid,  legend='Site #7',    color='orange',           line_width=2)
p1.line(Regal.index,       Regal.Rel_humid,       legend='Site #8',         color='khaki',            line_width=2)
p1.line(Sheridan.index,    Sheridan.Rel_humid,    legend='Site #9',      color='deepskyblue',      line_width=2)
p1.line(Stevens.index,     Stevens.Rel_humid,     legend='Site #10',       color='grey',             line_width=2)
#p1.line(Reference.index,  Reference.Rel_humid,   legend='Reference',    color='olive',             line_width=2)
#p1.line(Paccar.index,     Paccar.Rel_humid,      legend='Paccar',        color='lime',             line_width=2)


p1.legend.click_policy="hide"


tab1 = Panel(child=p1, title="Relative Humidity")

tabs = Tabs(tabs=[ tab1])

show(tabs)


#%%

# Comparison Data for indoor PMS5003 unit and Clarity Unit overlap for Highest 6 schools

# read in data from indoor units for 6 highest schools
# choose /resample to load in indoor PMS5003 data that has been resampled to 15 min for quicker loading
# Note that periodically must use /WSU (to get all the new csv files) to update the 15 min resample to get the most current set of data
# make sure to move the files in "resample backup" folder back into the "data" urbanova folders when r syncing so don't copy in all files from Rpi's again
# (and then take out all files that have already been resampled so don't have to spend 2 hours loading all just to cut off by dates and resample)

#date_range = '2_9_to_8_27_20'

grant = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Grant/resample*.csv')
files.sort()
for file in files:
    grant = pd.concat([grant, pd.read_csv(file)], sort=False)


stevens = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Stevens/resample*.csv')
files.sort()
for file in files:
    stevens = pd.concat([stevens, pd.read_csv(file)], sort=False)


balboa = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Balboa/resample*.csv')
files.sort()
for file in files:
    balboa = pd.concat([balboa, pd.read_csv(file)], sort=False)


adams = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Adams/resample*.csv')
files.sort()
for file in files:
    adams = pd.concat([adams, pd.read_csv(file)], sort=False)


jefferson = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Jefferson/resample*.csv')
files.sort()
for file in files:
    jefferson = pd.concat([jefferson, pd.read_csv(file)], sort=False)


sheridan = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Sheridan/resample*.csv')
files.sort()
for file in files:
    sheridan = pd.concat([sheridan, pd.read_csv(file)], sort=False)


# Comparison Data for indoor PMS5003 unit and Clarity Unit overlap for lowest 4 Clarity sensors

browne = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Browne/resample*.csv')
files.sort()
for file in files:
    browne = pd.concat([browne, pd.read_csv(file)], sort=False)


audubon = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Audubon/resample*.csv')
files.sort()
for file in files:
    audubon = pd.concat([audubon, pd.read_csv(file)], sort=False)


lidgerwood = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Lidgerwood/resample*.csv')
files.sort()
for file in files:
    lidgerwood = pd.concat([lidgerwood, pd.read_csv(file)], sort=False)


regal = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Regal/resample*.csv')
files.sort()
for file in files:
    regal = pd.concat([regal, pd.read_csv(file)], sort=False)


#%%

# only used to resample indoor data to lower frequency so doesnt take so long to load in each time
    
grant['Datetime'] = pd.to_datetime(grant['Datetime'])
grant = grant.sort_values('Datetime')

place_holder_times = grant.Datetime
del grant['Datetime']
grant = grant.astype(np.float64)
grant['Datetime'] = place_holder_times

grant.index = grant.Datetime
grant = grant.loc[start_time:end_time]
grant = grant.resample(interval).mean()
grant['Datetime'] = grant.index
grant['Location'] = 'Grant'


stevens['Datetime'] = pd.to_datetime(stevens['Datetime'])
stevens = stevens.sort_values('Datetime')

place_holder_times = stevens.Datetime
del stevens['Datetime']
stevens = stevens.astype(np.float64)
stevens['Datetime'] = place_holder_times

stevens.index = stevens.Datetime
stevens = stevens.loc[start_time:end_time] 
stevens = stevens.resample(interval).mean()
stevens['Datetime'] = stevens.index
stevens['Location'] = 'Stevens'



balboa['Datetime'] = pd.to_datetime(balboa['Datetime'])
balboa = balboa.sort_values('Datetime')

place_holder_times = balboa.Datetime
del balboa['Datetime']
balboa = balboa.astype(np.float64)
balboa['Datetime'] = place_holder_times

balboa.index = balboa.Datetime
balboa = balboa.loc[start_time:end_time] 
balboa = balboa.resample(interval).mean()
balboa['Datetime'] = balboa.index
balboa['Location'] = 'Balboa'


adams['Datetime'] = pd.to_datetime(adams['Datetime'])
adams = adams.sort_values('Datetime')

place_holder_times = adams.Datetime
del adams['Datetime']
adams = adams.astype(np.float64)
adams['Datetime'] = place_holder_times

adams.index = adams.Datetime
adams = adams.loc[start_time:end_time] 
adams = adams.resample(interval).mean()
adams['Datetime'] = adams.index
adams['Location'] = 'Adams'


jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson = jefferson.sort_values('Datetime')

place_holder_times = jefferson.Datetime
del jefferson['Datetime']
jefferson = jefferson.astype(np.float64)
jefferson['Datetime'] = place_holder_times

jefferson.index = jefferson.Datetime
jefferson = jefferson.loc[start_time:end_time] 
jefferson = jefferson.resample(interval).mean()
jefferson['Datetime'] = jefferson.index
jefferson['Location'] = 'Jefferson'


sheridan['Datetime'] = pd.to_datetime(sheridan['Datetime'])
sheridan = sheridan.sort_values('Datetime')

place_holder_times = sheridan.Datetime
del sheridan['Datetime']

sheridan = sheridan.astype(np.float64)
sheridan['Datetime'] = place_holder_times

sheridan.index = sheridan.Datetime
sheridan = sheridan.loc[start_time:end_time] 
sheridan = sheridan.resample(interval).mean()
sheridan['Datetime'] = sheridan.index
sheridan['Location'] = 'Sheridan'


browne['Datetime'] = pd.to_datetime(browne['Datetime'])
browne = browne.sort_values('Datetime')

place_holder_times = browne.Datetime
del browne['Datetime']
browne = browne.astype(np.float64)
browne['Datetime'] = place_holder_times

browne.index = browne.Datetime
browne = browne.loc[start_time:end_time] 
browne = browne.resample(interval).mean()
browne['Datetime'] = browne.index
browne['Location'] = 'Browne'


audubon['Datetime'] = pd.to_datetime(audubon['Datetime'])
audubon = audubon.sort_values('Datetime')

place_holder_times = audubon.Datetime
del audubon['Datetime']
audubon = audubon.astype(np.float64)
audubon['Datetime'] = place_holder_times

audubon.index = audubon.Datetime
audubon = audubon.loc[start_time:end_time] 
audubon = audubon.resample(interval).mean()
audubon['Datetime'] = audubon.index
audubon['Location'] = 'Audubon'


lidgerwood['Datetime'] = pd.to_datetime(lidgerwood['Datetime'])
lidgerwood = lidgerwood.sort_values('Datetime')

place_holder_times = lidgerwood.Datetime
del lidgerwood['Datetime']
lidgerwood = lidgerwood.astype(np.float64)
lidgerwood['Datetime'] = place_holder_times

lidgerwood.index = lidgerwood.Datetime
lidgerwood = lidgerwood.loc[start_time:end_time] 
lidgerwood = lidgerwood.resample(interval).mean()
lidgerwood['Datetime'] = lidgerwood.index
lidgerwood['Location'] = 'Lidgerwood'


regal['Datetime'] = pd.to_datetime(regal['Datetime'])
regal = regal.sort_values('Datetime')

place_holder_times = regal.Datetime
del regal['Datetime']
regal = regal.astype(np.float64)
regal['Datetime'] = place_holder_times

regal.index = regal.Datetime
regal = regal.loc[start_time:end_time] 
regal = regal.resample(interval).mean()
regal['Datetime'] = regal.index
regal['Location'] = 'Regal'

#%%

# just used to resample indoor data to lower frequency so doesnt take so long to load in each time
#date_range = '9_21_to_9_28_20'

#audubon.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Audubon/resample_15_min_audubon' + '_' + date_range + '.csv', index=False)
#adams.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Adams/resample_15_min_adams' + '_' + date_range + '.csv', index=False)
#balboa.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Balboa/resample_15_min_balboa' + '_' + date_range + '.csv', index=False)
#browne.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Browne/resample_15_min_browne' + '_' + date_range + '.csv', index=False)
#grant.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Grant/resample_15_min_grant' + '_' + date_range + '.csv', index=False)
#jefferson.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Jefferson/resample_15_min_jefferson' + '_' + date_range + '.csv', index=False)
#lidgerwood.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Lidgerwood/resample_15_min_lidgerwood' + '_' + date_range + '.csv', index=False)
#regal.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Regal/resample_15_min_regal' + '_' + date_range + '.csv', index=False)
#sheridan.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Sheridan/resample_15_min_sheridan' + '_' + date_range + '.csv', index=False)
#stevens.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Stevens/resample_15_stevens' + '_' + date_range + '.csv', index=False)


#%%
# Load in BME 280 data from running in seperate script after changed PMS5003 to default sampling rate
# only use for loading in new BME280 to add to resample files (those files are loaded in above at top of script)
# if load this, will only load 15 min resamples and might have less data (and it doesnt take too long to load in without the resample anyways, as done at the top of script because BME280 frequency slower than PMS5003)


audubon_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Audubon/15*.csv')
files.sort()
for file in files:
    audubon_bme = pd.concat([audubon_bme, pd.read_csv(file)], sort=False)

audubon_bme['Datetime'] = pd.to_datetime(audubon_bme['Datetime'])
audubon_bme = audubon_bme.sort_values('Datetime')
audubon_bme.index = audubon_bme.Datetime
audubon_bme = audubon_bme.loc[start_time:end_time] 
audubon_bme = audubon_bme.resample(interval).mean()
audubon_bme['Datetime'] = audubon_bme.index

adams_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Adams/15*.csv')
files.sort()
for file in files:
    adams_bme = pd.concat([adams_bme, pd.read_csv(file)], sort=False)
    
adams_bme['Datetime'] = pd.to_datetime(adams_bme['Datetime'])
adams_bme = adams_bme.sort_values('Datetime')
adams_bme.index = adams_bme.Datetime
adams_bme = adams_bme.loc[start_time:end_time] 
adams_bme = adams_bme.resample(interval).mean()
adams_bme['Datetime'] = adams_bme.index

balboa_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Balboa/15*.csv')
files.sort()
for file in files:
    balboa_bme = pd.concat([balboa_bme, pd.read_csv(file)], sort=False)
    
balboa_bme['Datetime'] = pd.to_datetime(balboa_bme['Datetime'])
balboa_bme = balboa_bme.sort_values('Datetime')
balboa_bme.index = balboa_bme.Datetime
balboa_bme = balboa_bme.loc[start_time:end_time] 
balboa_bme = balboa_bme.resample(interval).mean()
balboa_bme['Datetime'] = balboa_bme.index

browne_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Browne/15*.csv')
files.sort()
for file in files:
    browne_bme = pd.concat([browne_bme, pd.read_csv(file)], sort=False)
    
browne_bme['Datetime'] = pd.to_datetime(browne_bme['Datetime'])
browne_bme = browne_bme.sort_values('Datetime')
browne_bme.index = browne_bme.Datetime
browne_bme = browne_bme.loc[start_time:end_time] 
browne_bme = browne_bme.resample(interval).mean()
browne_bme['Datetime'] = browne_bme.index

grant_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Grant/15*.csv')
files.sort()
for file in files:
    grant_bme = pd.concat([grant_bme, pd.read_csv(file)], sort=False)
    
grant_bme['Datetime'] = pd.to_datetime(grant_bme['Datetime'])
grant_bme = grant_bme.sort_values('Datetime')
grant_bme.index = grant_bme.Datetime
grant_bme = grant_bme.loc[start_time:end_time] 
grant_bme = grant_bme.resample(interval).mean()
grant_bme['Datetime'] = grant_bme.index

jefferson_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Jefferson/15*.csv')
files.sort()
for file in files:
    jefferson_bme = pd.concat([jefferson_bme, pd.read_csv(file)], sort=False)
    
jefferson_bme['Datetime'] = pd.to_datetime(jefferson_bme['Datetime'])
jefferson_bme = jefferson_bme.sort_values('Datetime')
jefferson_bme.index = jefferson_bme.Datetime
jefferson_bme = jefferson_bme.loc[start_time:end_time] 
jefferson_bme = jefferson_bme.resample(interval).mean()
jefferson_bme['Datetime'] = jefferson_bme.index

lidgerwood_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Lidgerwood/15*.csv')
files.sort()
for file in files:
    lidgerwood_bme = pd.concat([lidgerwood_bme, pd.read_csv(file)], sort=False)
    
lidgerwood_bme['Datetime'] = pd.to_datetime(lidgerwood_bme['Datetime'])
lidgerwood_bme = lidgerwood_bme.sort_values('Datetime')
lidgerwood_bme.index = lidgerwood_bme.Datetime
lidgerwood_bme = lidgerwood_bme.loc[start_time:end_time] 
lidgerwood_bme = lidgerwood_bme.resample(interval).mean()
lidgerwood_bme['Datetime'] = lidgerwood_bme.index

regal_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Regal/15*.csv')
files.sort()
for file in files:
    regal_bme = pd.concat([regal_bme, pd.read_csv(file)], sort=False)
    
regal_bme['Datetime'] = pd.to_datetime(regal_bme['Datetime'])
regal_bme = regal_bme.sort_values('Datetime')
regal_bme.index = regal_bme.Datetime
regal_bme = regal_bme.loc[start_time:end_time] 
regal_bme = regal_bme.resample(interval).mean()
regal_bme['Datetime'] = regal_bme.index

sheridan_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Sheridan/15*.csv')
files.sort()
for file in files:
    sheridan_bme = pd.concat([sheridan_bme, pd.read_csv(file)], sort=False)
    
sheridan_bme['Datetime'] = pd.to_datetime(sheridan_bme['Datetime'])
sheridan_bme = sheridan_bme.sort_values('Datetime')
sheridan_bme.index = sheridan_bme.Datetime
sheridan_bme = sheridan_bme.loc[start_time:end_time] 
sheridan_bme = sheridan_bme.resample(interval).mean()
sheridan_bme['Datetime'] = sheridan_bme.index

stevens_bme = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Stevens/15*.csv')
files.sort()
for file in files:
    stevens_bme = pd.concat([stevens_bme, pd.read_csv(file)], sort=False)
    
stevens_bme['Datetime'] = pd.to_datetime(stevens_bme['Datetime'])
stevens_bme = stevens_bme.sort_values('Datetime')
stevens_bme.index = stevens_bme.Datetime
stevens_bme = stevens_bme.loc[start_time:end_time] 
stevens_bme = stevens_bme.resample(interval).mean()
stevens_bme['Datetime'] = stevens_bme.index

#%%

# just used to resample indoor data to lower frequency so doesnt take so long to load in each time
#date_range = '2_9_to_6_22_20'

#audubon_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Audubon/15_min_resample_audubon' + '_' + date_range + '.csv', index=False)
#adams_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Adams/15_min_resample_adams' + '_' + date_range + '.csv', index=False)
#balboa_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Balboa/15_min_resample_balboa' + '_' + date_range + '.csv', index=False)
#browne_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Browne/15_min_resample_browne' + '_' + date_range + '.csv', index=False)
#grant_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Grant/15_min_resample_grant' + '_' + date_range + '.csv', index=False)
#jefferson_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Jefferson/15_min_resample_jefferson' + '_' + date_range + '.csv', index=False)
#lidgerwood_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Lidgerwood/15_min_resample_lidgerwood' + '_' + date_range + '.csv', index=False)
#regal_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Regal/15_min_resample_regal' + '_' + date_range + '.csv', index=False)
#sheridan_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Sheridan/15_min_resample_sheridan' + '_' + date_range + '.csv', index=False)
#stevens_bme.to_csv('/Users/matthew/Desktop/data/urbanova/ramboll/Stevens/15_min_resample_stevens' + '_' + date_range + '.csv', index=False)

#%%

# add temp and relative humidity columns from BME280 to indoor unit PM 2.5 df
adams['Datetime'] = pd.to_datetime(adams['Datetime'])
adams.index = adams.Datetime
adams = adams.loc[start_time:end_time]
adams = adams.resample(interval).mean() 

adams['PM2_5_corrected'] = adams['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
adams['Rel_humid'] = adams_bme['RH']
adams['temp'] = adams_bme['temp']

audubon['Datetime'] = pd.to_datetime(audubon['Datetime'])
audubon.index = audubon.Datetime
audubon = audubon.loc[start_time:end_time]
audubon = audubon.resample(interval).mean() 

audubon['PM2_5_corrected'] = audubon['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
audubon['Rel_humid'] = audubon_bme['RH']
audubon['temp'] = audubon_bme['temp']


balboa['Datetime'] = pd.to_datetime(balboa['Datetime'])
balboa.index = balboa.Datetime
balboa = balboa.loc[start_time:end_time]
balboa = balboa.resample(interval).mean() 

balboa['PM2_5_corrected'] = balboa['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
balboa['Rel_humid'] = balboa_bme['RH']
balboa['temp'] = balboa_bme['temp']


browne['Datetime'] = pd.to_datetime(browne['Datetime'])
browne.index = browne.Datetime
browne = browne.loc[start_time:end_time]
browne = browne.resample(interval).mean() 

browne['PM2_5_corrected'] = browne['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
browne['Rel_humid'] = browne_bme['RH']
browne['temp'] = browne_bme['temp']


grant['Datetime'] = pd.to_datetime(grant['Datetime'])
grant.index = grant.Datetime
grant = grant.loc[start_time:end_time]
grant = grant.resample(interval).mean() 

grant['PM2_5_corrected'] = grant['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
grant['Rel_humid'] = grant_bme['RH']
grant['temp'] = grant_bme['temp']


jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson.index = jefferson.Datetime
jefferson = jefferson.loc[start_time:end_time]
jefferson = jefferson.resample(interval).mean() 

jefferson['PM2_5_corrected'] = jefferson['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
jefferson['Rel_humid'] = jefferson_bme['RH']
jefferson['temp'] = jefferson_bme['temp']


lidgerwood['Datetime'] = pd.to_datetime(lidgerwood['Datetime'])
lidgerwood.index = lidgerwood.Datetime
lidgerwood = lidgerwood.loc[start_time:end_time]
lidgerwood = lidgerwood.resample(interval).mean() 

lidgerwood['PM2_5_corrected'] = lidgerwood['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
lidgerwood['Rel_humid'] = lidgerwood_bme['RH']
lidgerwood['temp'] = lidgerwood_bme['temp']


regal['Datetime'] = pd.to_datetime(regal['Datetime'])
regal.index = regal.Datetime
regal = regal.loc[start_time:end_time]
regal = regal.resample(interval).mean() 

regal['PM2_5_corrected'] = regal['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
regal['Rel_humid'] = regal_bme['RH']
regal['temp'] = regal_bme['temp']


sheridan['Datetime'] = pd.to_datetime(sheridan['Datetime'])
sheridan.index = sheridan.Datetime
sheridan = sheridan.loc[start_time:end_time]
sheridan = sheridan.resample(interval).mean() 

sheridan['PM2_5_corrected'] = sheridan['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
sheridan['Rel_humid'] = sheridan_bme['RH']
sheridan['temp'] = sheridan_bme['temp']


stevens['Datetime'] = pd.to_datetime(stevens['Datetime'])
stevens.index = stevens.Datetime
stevens = stevens.loc[start_time:end_time]
stevens = stevens.resample(interval).mean() 

stevens['PM2_5_corrected'] = stevens['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
stevens['Rel_humid'] = stevens_bme['RH']
stevens['temp'] = stevens_bme['temp']

audubon['Location'] = 'Audubon'
adams['Location'] = 'Adams'
balboa['Location'] = 'Balboa'
browne['Location'] = 'Browne'
grant['Location'] = 'Grant'
jefferson['Location'] = 'Jefferson'
lidgerwood['Location'] = 'Lidgerwood'
regal['Location'] = 'Regal'
sheridan['Location'] = 'Sheridan'
stevens['Location'] = 'Stevens'


#%%

# Add specific humidity to indoor units so can use in MLR calibration
# Note that Jefferson, Grant, and Lidgerwood are using other locations (extremely close Pressure values though) because of missing data for when their BME's weren't running
    # This doesn't actually do anything because Jefferson and Grant was missing PM 2.5 data during that time and Lidgerwood is still missing temp and rh data which isn't as similar to other sites...
spec_humid(audubon_bme, audubon_bme_json, audubon)
spec_humid(adams_bme, adams_bme_json, adams)
spec_humid(balboa_bme, balboa_bme_json, balboa)
spec_humid(browne_bme, browne_bme_json, browne)
spec_humid(regal_bme, regal_bme_json, grant)             # note Grant using Regal BME
spec_humid(adams_bme, adams_bme_json, jefferson)              # note Jefferson using Adams BME
spec_humid(balboa_bme, balboa_bme_json, lidgerwood)  # note Lidgerwood using Balboa BME
spec_humid(regal_bme, regal_bme_json, regal)
spec_humid(sheridan_bme, sheridan_bme_json, sheridan)
spec_humid(stevens_bme, stevens_bme_json, stevens)

#%%
# apply mlr calibration from Augusta

mlr_function(mlr_model, adams)
mlr_function(mlr_model, audubon)
mlr_function(mlr_model, balboa)
mlr_function(mlr_model, browne)
mlr_function(mlr_model, grant)
mlr_function(mlr_model, jefferson)
mlr_function(mlr_model, lidgerwood)
mlr_function(mlr_model, regal)
mlr_function(mlr_model, sheridan)
mlr_function(mlr_model, stevens)


#%%

# add uncertainty bars

indoor_mlr_uncertainty(stdev_number,audubon)
indoor_mlr_uncertainty(stdev_number,adams)
indoor_mlr_uncertainty(stdev_number,balboa)
indoor_mlr_uncertainty(stdev_number,browne)
indoor_mlr_uncertainty(stdev_number,grant)
indoor_mlr_uncertainty(stdev_number,jefferson)
indoor_mlr_uncertainty(stdev_number,lidgerwood)
indoor_mlr_uncertainty(stdev_number,regal)
indoor_mlr_uncertainty(stdev_number,sheridan)
indoor_mlr_uncertainty(stdev_number,stevens)


#%%
#checking function output
#balboa['out_PM2_5_corrected'] = Balboa['PM2_5_corrected']
#balboa['PM2_5_corrected'] = balboa['PM2_5_corrected'].shift(-1)

#%%

# calculate indoor/outdoor correlation at each location (note that this acutally shifts the indoor dataframes as well!!! so make sure to reset after calling this)
# Should recalc this each time that the overall dataset being looked at changes

corr_df = pd.DataFrame({})
corr_df['offset'] = [0,-1,-2,-3,-4,-5, -6 , -7, -8, -9 , -10, -11, -12]
    
in_out_corr(corr_df, audubon, Audubon)
in_out_corr(corr_df, adams, Adams)
in_out_corr(corr_df, balboa, Balboa)
in_out_corr(corr_df, browne, Browne)
in_out_corr(corr_df, grant, Grant)
in_out_corr(corr_df, jefferson, Jefferson)
in_out_corr(corr_df, lidgerwood, Lidgerwood)
in_out_corr(corr_df, regal, Regal)
in_out_corr(corr_df, sheridan, Sheridan)
in_out_corr(corr_df, stevens, Stevens)

#%%

# shift indoor units according to their optimal r value
# make sure to just run this once, if do it more the data will keep being shifted

opt_shift(audubon, corr_df)
opt_shift(grant, corr_df)
opt_shift(regal, corr_df)
opt_shift(sheridan, corr_df)
opt_shift(stevens, corr_df)
opt_shift(adams, corr_df)
opt_shift(balboa, corr_df)
opt_shift(browne, corr_df)
opt_shift(jefferson, corr_df)
opt_shift(lidgerwood, corr_df)

#%%

# Create empty df's to initiate function (the columns are passed to lists that get appended and the df gets wiped and rebuilt each run so the index matches each loop)


df_threshold_1 = pd.DataFrame({})
df_threshold_2 = pd.DataFrame({})
df_threshold_3 = pd.DataFrame({})
df_threshold_4 = pd.DataFrame({}) #index=range(number_locations)

df_list = [df_threshold_1, df_threshold_2, df_threshold_3, df_threshold_4]


for i in df_list:
    i['Location'] = None
    i['avg_fraction_filtered'] = None
    i['out_avg'] = None
    i['in_avg'] = None
    i['out_med'] = None
    i['in_med'] = None
    i['count'] = None
    i['percentage_total_measurements'] = None

#%%
# plot histogram of calibrated indoor PM2.5 - shifted calibrated indoor PM2.5
# Note that if using threshold of 0, 5, 10 ,15, Adams and Audubon error out on the 15 threshold because they dont have any outdoor values above 15

# So for Audubon and Adams, only use up to df_threshold_3, and go into in_out_histogram functiona and limit to 2 thresholds

df_list = in_out_histogram(audubon, Audubon, df_list)
#%%
df_list = in_out_histogram(adams, Adams, df_list)
#%%
df_list = in_out_histogram(balboa, Balboa, df_list)

df_list = in_out_histogram(browne, Browne, df_list)

df_list = in_out_histogram(grant, Grant,df_list)

df_list = in_out_histogram(jefferson, Jefferson, df_list)

df_list = in_out_histogram(lidgerwood, Lidgerwood, df_list)

df_list = in_out_histogram(regal, Regal, df_list)

df_list = in_out_histogram(sheridan, Sheridan, df_list)

df_list = in_out_histogram(stevens, Stevens, df_list)


#%%
indoor_outdoor_plot(adams, Adams)
#%%
indoor_outdoor_plot(audubon, Audubon)
indoor_outdoor_plot(balboa, Balboa)
indoor_outdoor_plot(browne, Browne)
indoor_outdoor_plot(grant, Grant)
indoor_outdoor_plot(jefferson, Jefferson)
indoor_outdoor_plot(lidgerwood, Lidgerwood)
indoor_outdoor_plot(regal, Regal)
indoor_outdoor_plot(sheridan, Sheridan)
indoor_outdoor_plot(stevens, Stevens)



#%%

p1 = figure(title = 'Grant',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(grant.index,     grant.PM2_5_corrected,  legend='Indoor PMS5003',  muted_color='red', muted_alpha=0.2,     color='red',     line_width=2)
#p1.line(Paccar.index,        Paccar.PM2_5,              legend='Clarity',       color='blue',      line_width=2) 
p1.line(Grant.index,     Grant.PM2_5_corrected,           legend='Outside Calibrated',  muted_color='black', muted_alpha=0.4,  line_alpha=0.4,   color='black',       line_width=2) 
p1.line(grant.index,     grant.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)

source_error = ColumnDataSource(data=dict(base=Grant.index, lower=Grant.lower_uncertainty, upper=Grant.upper_uncertainty))

#p1.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)

p1.legend.click_policy="mute"
p1.legend.location='top_left'
    
p2 = figure(title = 'Stevens',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p2.line(stevens.index,     stevens.PM2_5_corrected,  legend='Indoor PMS5003',       color='red',  muted_color='red', muted_alpha=0.2,   line_width=2)
p2.line(Stevens.index,     Stevens.PM2_5_corrected,           legend='Outside Calibrated',  muted_color='black', muted_alpha=0.4, line_alpha=0.4,    color='black',       line_width=2) 
source_error = ColumnDataSource(data=dict(base=Stevens.index, lower=Stevens.lower_uncertainty, upper=Stevens.upper_uncertainty))
p2.line(stevens.index,     stevens.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)


#p2.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)

p2.legend.click_policy="mute"
p2.legend.location='top_left'
    
p3 = figure(title = 'Balboa',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p3.line(balboa.index,     balboa.PM2_5_corrected,  legend='Indoor PMS5003',       color='red', muted_color='red', muted_alpha=0.2,    line_width=2)
p3.line(Balboa.index,     Balboa.PM2_5_corrected,           legend='Outside Calibrated',   muted_color='black', muted_alpha=0.4, line_alpha=0.4,   color='black',       line_width=2) 
source_error = ColumnDataSource(data=dict(base=Balboa.index, lower=Balboa.lower_uncertainty, upper=Balboa.upper_uncertainty))
p3.line(balboa.index,     balboa.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)


#p3.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p3.legend.click_policy="mute"
p3.legend.location='top_left'
    
p4 = figure(title = 'Adams',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p4.line(adams.index,     adams.PM2_5_corrected,  legend='Indoor PMS5003',       color='red',   muted_color='red', muted_alpha=0.2,  line_width=2)
p4.line(Adams.index,     Adams.PM2_5_corrected,           legend='Outside Calibrated',       color='black',  muted_color='black', muted_alpha=0.4, line_alpha=0.4,    line_width=2) 
source_error = ColumnDataSource(data=dict(base=Adams.index, lower=Adams.lower_uncertainty, upper=Adams.upper_uncertainty))
p4.line(adams.index,     adams.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)


#p4.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p4.legend.click_policy="mute"
p4.legend.location='top_left'


p5 = figure(title = 'Jefferson',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p5.line(jefferson.index,     jefferson.PM2_5_corrected,  legend='Indoor PMS5003',  muted_color='red', muted_alpha=0.2,     color='red',     line_width=2)
p5.line(Jefferson.index,     Jefferson.PM2_5_corrected,           legend='Outside Calibrated',    muted_color='black', muted_alpha=0.4,   color='black', line_alpha=0.4,      line_width=2) 
p5.line(Reference.index,     Reference.PM2_5_corrected,        legend = 'Clarity Reference',       color='blue',  muted_color='blue', muted_alpha=0.2  ,   line_width=2)
source_error = ColumnDataSource(data=dict(base=Jefferson.index, lower=Jefferson.lower_uncertainty, upper=Jefferson.upper_uncertainty))
p5.line(jefferson.index,     jefferson.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)


#tab1 = Panel(child=p5, title="Indoor Outdoor Comparison")

#tabs = Tabs(tabs=[ tab1])

#show(tabs)

#p5.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p5.legend.click_policy="mute"
p5.legend.location='top_left'

p6 = figure(title = 'Sheridan',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p6.line(sheridan.index,     sheridan.PM2_5_corrected,  legend='Indoor PMS5003',       color='red',  muted_color='red', muted_alpha=0.2,   line_width=2)
p6.line(Sheridan.index,     Sheridan.PM2_5_corrected,           legend='Outside Calibrated',   muted_color='black', muted_alpha=0.4, line_alpha=0.4,   color='black',       line_width=2) 
source_error = ColumnDataSource(data=dict(base=Sheridan.index, lower=Sheridan.lower_uncertainty, upper=Sheridan.upper_uncertainty))
p6.line(sheridan.index,     sheridan.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)


#p6.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p6.legend.click_policy="mute"
p6.legend.location='top_left'

p7 = gridplot([[p1,p2, p3], [p4, p5, p6]], plot_width = 400, plot_height = 300)

export_png(p7, filename='/Users/matthew/Desktop/data/Indoor_Outdoor_Upper_Schools_Comparison.png')

tab1 = Panel(child=p7, title="Indoor Outdoor Comparison")

tabs = Tabs(tabs=[ tab1])

show(tabs)


#%%

p1 = figure(title = 'Browne',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(browne.index,     browne.PM2_5_corrected,  legend='Indoor PMS5003',       color='red',  muted_color='red', muted_alpha=0.2,   line_width=2)
p1.line(Browne.index,     Browne.PM2_5_corrected,           legend='Outside Calibrated',       color='black', muted_color='black', muted_alpha=0.4,  line_alpha=0.4,    line_width=2) 
source_error = ColumnDataSource(data=dict(base=Browne.index, lower=Browne.lower_uncertainty, upper=Browne.upper_uncertainty))
p1.line(browne.index,     browne.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)

#p1.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p1.legend.click_policy="mute"
p1.legend.location='top_left'


p2 = figure(title = 'Audubon',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p2.line(audubon.index,     audubon.PM2_5_corrected,  legend='Indoor PMS5003',       color='red',  muted_color='red', muted_alpha=0.2,   line_width=2)
p2.line(Audubon.index,     Audubon.PM2_5_corrected,           legend='Outside Calibrated',       color='black', muted_color='black', muted_alpha=0.4, line_alpha=0.4,     line_width=2) 
source_error = ColumnDataSource(data=dict(base=Audubon.index, lower=Audubon.lower_uncertainty, upper=Audubon.upper_uncertainty))
p2.line(audubon.index,     audubon.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)

#p2.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p2.legend.click_policy="mute"
p2.legend.location='top_left'
    
p3 = figure(title = 'Lidgerwood',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p3.line(lidgerwood.index,     lidgerwood.PM2_5_corrected,  legend='Indoor PMS5003',       color='red', muted_color='red', muted_alpha=0.2,    line_width=2)
p3.line(Lidgerwood.index,     Lidgerwood.PM2_5_corrected,           legend='Outside Calibrated',       color='black',  muted_color='black', muted_alpha=0.4, line_alpha=0.4,    line_width=2) 
source_error = ColumnDataSource(data=dict(base=Lidgerwood.index, lower=Lidgerwood.lower_uncertainty, upper=Lidgerwood.upper_uncertainty))
p3.line(lidgerwood.index,     lidgerwood.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)

#p3.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p3.legend.click_policy="mute"
p3.legend.location='top_left'
    
p4 = figure(title = 'Regal',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p4.line(regal.index,     regal.PM2_5_corrected,  legend='Indoor PMS5003',       color='red', muted_color='red', muted_alpha=0.2,    line_width=2)
p4.line(Regal.index,     Regal.PM2_5_corrected,           legend='Outside Calibrated',       color='black',   muted_color='black', muted_alpha=0.4, line_alpha=0.4,   line_width=2) 
source_error = ColumnDataSource(data=dict(base=Regal.index, lower=Regal.lower_uncertainty, upper=Regal.upper_uncertainty))
p4.line(regal.index,     regal.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', muted_alpha=0.4,     color='blue',     line_width=2)

#p4.add_layout(
#    Whisker(source=source_error, base="base", upper="upper", lower="lower")
#)
p4.legend.click_policy="mute"
p4.legend.location='top_left'


p5 = gridplot([[p1,p2], [p3, p4]], plot_width = 500, plot_height = 300)

export_png(p5, filename='/Users/matthew/Desktop/data/Indoor_Outdoor_Lower_Schools_Comparison.png')

tab1 = Panel(child=p5, title="Indoor Outdoor Comparison")

tabs = Tabs(tabs=[ tab1])

show(tabs)




#%% 

# Plotting data from BME sensors

p1 = figure(title = 'Temperature',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='T (C)')

p1.line(adams_bme.index,          adams_bme.temp,       legend='Adams Indoor Temp',            color='gold',       line_width=2)        
p1.line(audubon_bme.index,        audubon_bme.temp,     legend='Audubon Indoor Temp',          color='yellow',     line_width=2)
p1.line(balboa_bme.index,         balboa_bme.temp,      legend='Balboa Indoor Temp',           color='gray',       line_width=2)
p1.line(browne_bme.index,         browne_bme.temp,      legend='Browne Indoor Temp',           color='orange',     line_width=2)
p1.line(grant_bme.index,          grant_bme.temp,       legend='Grant Indoor Temp',            color='purple',     line_width=2)
p1.line(jefferson_bme.index,      jefferson_bme.temp,   legend='Jefferson Indoor Temp',        color='blue',       line_width=2)
p1.line(lidgerwood_bme.index,     lidgerwood_bme.temp,  legend='Lidgerwood Indoor Temp',       color='green',      line_width=2)
p1.line(regal_bme.index,          regal_bme.temp,       legend='Regal Indoor Temp',            color='black',      line_width=2)
p1.line(sheridan_bme.index,       sheridan_bme.temp,    legend='Sheridan Indoor Temp',         color='red',        line_width=2)
p1.line(stevens_bme.index,        stevens_bme.temp,     legend='Stevens Indoor Temp',          color='blue',       line_width=2)
p1.legend.location='top_left'

p1.legend.click_policy="hide"
tab1 = Panel(child=p1, title="Temperature")

p2 = figure(title = 'Pressure',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='P (mb)')

p2.line(adams_bme.index,        adams_bme.P,       legend='Adams Indoor Pressure',         color='gold',       line_width=2)        
p2.line(audubon_bme.index,      audubon_bme.P,     legend='Audubon Indoor Pressure',       color='yellow',     line_width=2)
p2.line(balboa_bme.index,       balboa_bme.P,      legend='Balboa Indoor Pressure',        color='gray',       line_width=2)
p2.line(browne_bme.index,       browne_bme.P,      legend='Browne Indoor Pressure',        color='orange',     line_width=2)
p2.line(grant_bme.index,        grant_bme.P,       legend='Grant Indoor Pressure',         color='purple',     line_width=2)
p2.line(jefferson_bme.index,    jefferson_bme.P,   legend='Jefferson Indoor Pressure',     color='blue',       line_width=2)
p2.line(lidgerwood_bme.index,   lidgerwood_bme.P,  legend='Lidgerwood Indoor Pressure',    color='green',      line_width=2)
p2.line(regal_bme.index,        regal_bme.P,       legend='Regal Indoor Pressure',         color='black',      line_width=2)
p2.line(sheridan_bme.index,     sheridan_bme.P,    legend='Sheridan Indoor Pressure',       color='red',        line_width=2)
p2.line(stevens_bme.index,      stevens_bme.P,     legend='Stevens Indoor Pressure',        color='blue',       line_width=2)
p2.legend.location='top_left'

p2.legend.click_policy="hide"
tab2 = Panel(child=p2, title="Pressure")

p3 = figure(title = 'Temperature',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')

p3.line(adams_bme.index,       adams_bme.RH,       legend='Adams Indoor Puressure',            color='gold',     line_width=2)        
p3.line(audubon_bme.index,     audubon_bme.RH,     legend='Audubon Indoor Puressure',          color='yellow',   line_width=2)
p3.line(balboa_bme.index,      balboa_bme.RH,      legend='Balboa Indoor Puressure',           color='gray',     line_width=2)
p3.line(browne_bme.index,      browne_bme.RH,      legend='Browne Indoor Puressure',           color='orange',   line_width=2)
p3.line(grant_bme.index,       grant_bme.RH,       legend='Grant Indoor Puressure',            color='purple',   line_width=2)
p3.line(jefferson_bme.index,   jefferson_bme.RH,   legend='Jefferson Indoor Puressure',        color='blue',     line_width=2)
p3.line(lidgerwood_bme.index,  lidgerwood_bme.RH,  legend='Lidgerwood Indoor Puressure',       color='green',    line_width=2)
p3.line(regal_bme.index,       regal_bme.RH,       legend='Regal Indoor Puressure',            color='black',    line_width=2)        
p3.line(sheridan_bme.index,    sheridan_bme.RH,     legend='Sheridan Indoor RH',                color='red',      line_width=2)
p3.line(stevens_bme.index,     stevens_bme.RH,     legend='Stevens Indoor RH',                 color='blue',     line_width=2)

p3.legend.location='top_left'
p3.legend.click_policy="hide"

tab3 = Panel(child=p3, title="Relative Humidity")

tabs = Tabs(tabs=[ tab1, tab2, tab3])

show(tabs)
#%%
# Plot relative humidity and clarity raw PM 2.5 together on time series

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/Clarity_Rel_Humidity.html')

p1 = figure(title = 'Relative Humidity',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')

p1.line(Audubon.index,          Audubon.Rel_humid,       legend='Adams RH',            color='black',       line_width=2)        

p2 = figure(title = 'Audubon PM 2.5',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m^3)')

p2.line(Audubon.index,          Audubon.PM2_5,       legend='Adams PM 2.5',            color='black',       line_width=2)

p3 = gridplot([[p1],[p2]], plot_width = 500, plot_height = 300)

tab1 = Panel(child=p3, title="Audubon")
tabs = Tabs(tabs=[ tab1])

export_png(p3, filename='/Users/matthew/Desktop/data/Clarity_Rel_Humidity.png')

show(tabs)

#%%

#Plot SRCAA monitors together

# Plot relative humidity and clarity raw PM 2.5 together on time series

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/Ref_sites_compare_time_series.html')

p1 = figure(title = 'PM 2.5',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m^3)')

p1.line(Augusta.index,          Augusta.PM2_5,       legend='Augusta BAM PM 2.5',            color='black',       line_width=2)        
p1.line(Monroe.index,          Monroe.PM2_5,       legend='Monroe Neph PM 2.5',            color='red',       line_width=2)        
p1.line(Broadway.index,          Broadway.PM2_5,       legend='Broadway BAM PM 2.5',            color='green',       line_width=2)        
p1.line(Greenbluff.index,          Greenbluff.PM2_5,       legend='Greenbluff TEOM PM 2.5',      line_alpha=0.7,      color='gray',       line_width=2)        

tab1 = Panel(child=p1, title="Reference Comparison")
tabs = Tabs(tabs=[ tab1])

export_png(p1, filename='/Users/matthew/Desktop/data/Ref_sites_compare_time_series.png')

show(tabs)

#%%
# count number of measurements at Grant

p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='# Measurements (ug/m3)')
        
p2.line(Grant_count.index,     Grant_count.PM2_5,  legend='Grant',       color='blue',     line_width=2)
p2.line(Paccar_count.index,     Paccar_count.PM2_5,  legend='Paccar',       color='black',     line_width=2)

tab2 = Panel(child=p2, title="# of Measurements at Grant")

export_png(p2, filename="/Users/matthew/Desktop/number_of_Grant_measurements.png")

tabs = Tabs(tabs=[tab2])

show(tabs)


#%%
# Calculate average and median PM 2.5 measurements for each Clarity Node

Node = [Audubon,
        Adams,
        Balboa,
        Browne,
        Grant,
        Jefferson,
        Lidgerwood,
        Regal,
        Sheridan,
        Stevens]#,
    #    Reference,
    #    Paccar,
    #    Augusta,
    #    Monroe,
    #    Broadway,
    #    Greenbluff]

Node_name = ['Audubon',
             'Adams',
             'Balboa',
             'Browne',
             'Grant',
             'Jefferson',
             'Lidgerwood',
             'Regal',
             'Sheridan',
             'Stevens']#,
         #    'Reference',
         #    'Paccar',
         #    'Augusta',
         #    'Monroe',
         #    'Broadway',
         #    'Greenbluff']
#%%

#For checking loop
audubon_avg = np.mean(Audubon['PM2_5'])
adams_avg = np.mean(Adams['PM2_5'])
balboa_avg = np.mean(Balboa['PM2_5'])
browne_avg = np.mean(Browne['PM2_5'])
grant_avg = np.mean(Grant['PM2_5'])
jefferson_avg = np.mean(Jefferson['PM2_5'])
lidgerwood_avg = np.mean(Lidgerwood['PM2_5'])
regal_avg = np.mean(Regal['PM2_5'])
sheridan_avg = np.mean(Sheridan['PM2_5'])
stevens_avg = np.mean(Stevens['PM2_5'])
#reference_avg = np.mean(Reference['PM2_5'])
#paccar_avg = np.mean(Paccar['PM2_5'])


audubon_avg_adj = np.mean(Audubon['PM2_5_corrected'])
adams_avg_adj = np.mean(Adams['PM2_5_corrected'])
balboa_avg_adj = np.mean(Balboa['PM2_5_corrected'])
browne_avg_adj = np.mean(Browne['PM2_5_corrected'])
grant_avg_adj = np.mean(Grant['PM2_5_corrected'])
jefferson_avg_adj = np.mean(Jefferson['PM2_5_corrected'])
lidgerwood_avg_adj = np.mean(Lidgerwood['PM2_5_corrected'])
regal_avg_adj = np.mean(Regal['PM2_5_corrected'])
sheridan_avg_adj = np.mean(Sheridan['PM2_5_corrected'])
stevens_avg_adj = np.mean(Stevens['PM2_5_corrected'])
#reference_avg_adj = np.mean(Reference['PM2_5_corrected'])
#paccar_avg_adj = np.mean(Paccar['PM2_5_corrected'])


avg = []
median = []
#mode = []               # not using mode because the mode is below 1
var = []
stdev = []
std_err = []

for name in Node:
    avg.append(np.mean(name['PM2_5']))
    median.append(np.median(name['PM2_5']))
    #mode.append(stats.mode(name['PM2_5']))
    var.append(np.var(name['PM2_5']))
    stdev.append(np.std(name['PM2_5']))
    std_err.append(stats.sem(name['PM2_5']))

stats_table = pd.DataFrame()
stats_table['location'] = Node_name
stats_table['avg'] = avg
stats_table['median'] = median
#stats_table['mode'] = mode
stats_table['var'] = var
stats_table['stdev'] = stdev
stats_table['std_err'] = std_err

avg_adj = []
median_adj = []
#mode = []               # not using mode because the mode is below 1
var_adj = []
stdev_adj = []
std_err_adj = []


for name in Node:
    avg_adj.append(np.nanmean(name['PM2_5_corrected']))
    median_adj.append(np.nanmedian(name['PM2_5_corrected']))
    #mode_adj.append(stats.mode(name['PM2_5']))
    var_adj.append(np.var(name['PM2_5_corrected']))
    stdev_adj.append(np.std(name['PM2_5_corrected']))
    std_err_adj.append(stats.sem(name['PM2_5_corrected']))

stats_table_adj = pd.DataFrame()
stats_table_adj['location'] = Node_name
stats_table_adj['avg'] = avg_adj
stats_table_adj['median'] = median_adj
#stats_table_adj['mode'] = mode_adj
stats_table_adj['var'] = var_adj
stats_table_adj['stdev'] = stdev_adj
stats_table_adj['std_err'] = std_err_adj
stats_table_adj = stats_table_adj.sort_values('location')
stats_table_adj = stats_table_adj.reset_index(drop=True)

stats_table_adj.to_csv('/Users/matthew/Desktop//stats.csv', index=False)
#%%
# plot average values 

info = [[47.621172,   -117.367725,   adams_avg_adj,      'Adams'],
        [47.621533,   -117.4098417,  jefferson_avg_adj,  'Jefferson'],
        [47.6467083,  -117.390983,   grant_avg_adj,      'Grant'],
        [47.6522472,  -117.355561,   sheridan_avg_adj,   'Sheridan'],
     #   [47.6608,     -117.4045056,  reference_avg_adj,  'Reference'],
        [47.671256,   -117.3846583,  stevens_avg_adj,    'Stevens'],
        [47.69735,    -117.369972,   regal_avg_adj,      'Regal'],
        [47.6798472,  -117.441739,   audubon_avg_adj,    'Audubon'],
        [47.7081417,  -117.405161,   lidgerwood_avg_adj, 'Lidgerwood'],
        [47.70415,    -117.4640639,  browne_avg_adj,     'Browne'],
        [47.71818056, -117.4560056,  balboa_avg_adj,     'Balboa']]

df = pd.DataFrame(info, columns =['Lat', 'Lon', 'avg_PM2_5','Location'], dtype = float)
#%%

#### USE THIS ONE   #####


df = location_traits
df = df.sort_values('School')
df = df.reset_index(drop=True)
df['avg_PM2_5'] = stats_table_adj['avg']
df['stdev'] = stats_table_adj['stdev']

#%%
print(df.dtypes)
#%%
label_locations = [[6044100,     -13062900,   'Adams'],
                  [6044200,      -13073000,   'Jefferson'],
                  [6048400,      -13070000,   'Grant'],
                  [6048100,      -13062300,   'Sheridan'],
           #       [6050600,      -13073000,   'Reference'],
                  [6052300,      -13064300,   'Stevens'],
                  [6056700,      -13063400,   'Regal'],
                  [6053850,      -13076600,   'Audubon'],
                  [6059950,      -13068000,   'Lidgerwood'],
                  [6057850,      -13078600,   'Browne'],
                  [6061500,      -13075300,   'Balboa']]

df1 = pd.DataFrame(label_locations, columns =['Lat', 'Lon', 'School'], dtype = float)
df1 = df1.sort_values('School')
df1 = df1.reset_index(drop=True)
df1['Type_of_School'] = df['Type_of_School']

#%%
#scatter = hv.Scatter(df.dropna(), kdims='Lon', vdims=['Lat', 'avg_PM2_5', 'School'])
#scatter.opts(color='avg_PM2_5', size=10, padding=.1, tools=['hover'], colorbar=True, cmap='magma', width=500, height=400)#, clim=(0, 60))

points = gv.Points(df.dropna(), ['Lon', 'Lat'], ['avg_PM2_5', 'School'])
points.opts(size=10, color='avg_PM2_5', cmap='magma', tools=['hover'], colorbar=True, width=500, height=400, padding=.1)#, clim=(0, 60))

labels = hv.Labels(df1, kdims = ['Lon', 'Lat'], vdims =['Type_of_School'])

test = gvts.EsriImagery * points * labels



hv.save(test.options(toolbar=None), '/Users/matthew/Desktop/IDW_new_test.png', fmt='png', backend='bokeh')    # works #test3.options(toolbar=None),
show(hv.render(test))

#%%

p1 = figure(plot_width=1100,
            plot_height=450,
         #   x_axis_type='datetime',
            x_axis_label='Distance to Freeway (ft)',
            y_axis_label='Average PM 2.5 (ug/m3)')

p1.title.text = 'PM2.5 vs Distance to Freeway'    

p1.scatter(df.Distance_to_freeway,    df.avg_PM2_5,     color='black',   muted_color='black', muted_alpha=0.1 , line_width=0.2)


source = ColumnDataSource(data=dict(Distance_to_freeway = np.array(df.Distance_to_freeway),
                                    avg_PM2_5 = np.array(df.avg_PM2_5),
                                    School = np.array(df.School)))

labels = LabelSet(x='Distance_to_freeway', y='avg_PM2_5', text='School', level='glyph',
         x_offset=-4, y_offset=2, source=source)

p1.add_layout(labels)

p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="PM 2.5 vs Atm Conditions Time Series")



location = df.School
avg_PM2_5 = df.avg_PM2_5

p2 = figure(x_range=location, plot_height=250, title="Locations",
           toolbar_location=None, tools="")


p2.vbar(x=location, top=avg_PM2_5, width=0.9)

p2.x_range.range_padding = 0
p2.xgrid.grid_line_color = None
p2.y_range.start = 0

tab2 = Panel(child=p2, title="PM 2.5 vs Location")

tabs = Tabs(tabs=[ tab1,tab2])

show(tabs)

#%%

from bokeh.io import output_file, show
from bokeh.plotting import figure

output_file("bars.html")

fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
counts = [5, 3, 4, 2, 4, 6]

p = figure(x_range=fruits, plot_height=250, title="Fruit Counts",
           toolbar_location=None, tools="")

p.vbar(x=fruits, top=counts, width=0.9)

p.xgrid.grid_line_color = None
p.y_range.start = 0

show(p)
#%%
# USE 60 min resample for this stat

# Check for exceeding 24 hr average of 35 ug/m3

# make start data and end date for 35 ug/3 limit to graph

for name in Node:
    name['Moving_Avg'] = name['PM2_5_corrected'].rolling(window=24).mean()
    exceeded = name[name['Moving_Avg'] >= 35]
    exceeded = exceeded['Moving_Avg']
    name['exceeded'] = exceeded
    print(exceeded)

#%%
Audubon.to_csv('/Users/matthew/Desktop/Clarity_data/Audubon.csv', index=False)
Adams.to_csv('/Users/matthew/Desktop/Clarity_data/Adams.csv', index=False)
Balboa.to_csv('/Users/matthew/Desktop/Clarity_data/Balboa.csv', index=False)
Browne.to_csv('/Users/matthew/Desktop/Clarity_data/Browne.csv', index=False)
Grant.to_csv('/Users/matthew/Desktop/Clarity_data/Grant.csv', index=False)
Jefferson.to_csv('/Users/matthew/Desktop/Clarity_data/Jefferson.csv', index=False)
Lidgerwood.to_csv('/Users/matthew/Desktop/Clarity_data/Lidgerwood.csv', index=False)
Regal.to_csv('/Users/matthew/Desktop/Clarity_data/Regal.csv', index=False)
Sheridan.to_csv('/Users/matthew/Desktop/Clarity_data/Sheridan.csv', index=False)
Stevens.to_csv('/Users/matthew/Desktop/Clarity_data/Stevens.csv', index=False)
#%%
# Send data to Mark Rowe at SRCAA from BAM overlap
unit_1.to_csv('/Users/matthew/Desktop/Clarity_data/unit_1.csv', index=False , date_format='%Y-%m-%d %H:%M:%S')
unit_2.to_csv('/Users/matthew/Desktop/Clarity_data/unit_2.csv', index=False , date_format='%Y-%m-%d %H:%M:%S')


