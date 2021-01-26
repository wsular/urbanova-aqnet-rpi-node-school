#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:00:58 2020

@author: matthew
"""

import pandas as pd
from glob import glob
import numpy as np
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure

import holoviews as hv
hv.extension('bokeh', logo=False)



from Augusta_BAM_uncertainty import Augusta_BAM_uncertainty
from high_cal_mlr_function_generator import high_cal_setup, generate_mlr_function_high_cal
from high_cal_mlr_function import mlr_function_high_cal



#%%

# Plot all Clarity PM 2.5 using interactive legend (mute)

def plot(location,location_name):
    
    if PlotType=='notebook':
        output_notebook()
    else:
        output_file('/Users/matthew/Desktop/clarity_PM2_5_time_series_legend_mute.html')

    p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Breakpoint (ug/m^3)',
            y_axis_label='Negative Value Count')

    p1.title.text = 'Calibration Breakpoint Check'

    p1.line(location.Threshold,     location.Negative_Count,     legend=location_name,       color='black',       line_width=2, muted_color='green', muted_alpha=0.2)

    p1.legend.click_policy="mute"

    tab1 = Panel(child=p1, title="Calibration Breakpoint Check")

    tabs = Tabs(tabs=[ tab1])

    show(tabs)

#%%

# threshold for counting negative values
l = -5

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
start_time = '2019-09-01 07:00'
end_time = '2020-11-16 07:00'

# Date Range of interest
#start_time = '2020-09-10 07:00'   # was 2/9//20 7:00
#end_time = '2020-09-21 07:00'

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

Audubon_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Audubon*.csv')
files.sort()
for file in files:
    Audubon_All = pd.concat([Audubon_All, pd.read_csv(file)], sort=False)

Adams_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Adams*.csv')
files.sort()
for file in files:
    Adams_All = pd.concat([Adams_All, pd.read_csv(file)], sort=False)

Balboa_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Balboa*.csv')
files.sort()
for file in files:
    Balboa_All = pd.concat([Balboa_All, pd.read_csv(file)], sort=False)

Browne_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Browne*.csv')
files.sort()
for file in files:
    Browne_All = pd.concat([Browne_All, pd.read_csv(file)], sort=False)

# drop erroneous data from Nov. 2019 when sensor malfunctioning
Browne_All = Browne_All[Browne_All['PM2_5'] < 1000]


Grant_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Grant*.csv')
files.sort()
for file in files:
    Grant_All = pd.concat([Grant_All, pd.read_csv(file)], sort=False)

Jefferson_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Jefferson*.csv')
files.sort()
for file in files:
    Jefferson_All = pd.concat([Jefferson_All, pd.read_csv(file)], sort=False)

Lidgerwood_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Lidgerwood*.csv')
files.sort()
for file in files:
    Lidgerwood_All = pd.concat([Lidgerwood_All, pd.read_csv(file)], sort=False)

Regal_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Regal*.csv')
files.sort()
for file in files:
    Regal_All = pd.concat([Regal_All, pd.read_csv(file)], sort=False)
    
Sheridan_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Sheridan*.csv')
files.sort()
for file in files:
    Sheridan_All = pd.concat([Sheridan_All, pd.read_csv(file)], sort=False)

Stevens_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Stevens*.csv')
files.sort()
for file in files:
    Stevens_All = pd.concat([Stevens_All, pd.read_csv(file)], sort=False)

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



#%%

# winter mlr calibration for low concentrations

# using typed out equation rather than mlr function because function was returning entire column rather than just the 
# calibration performed on measurements less than 100 (ie was affecting the high measurements as well)
# note that all the constants are the same because the same mlr equation derived from the Clarity reference node at 
# the Augusta site is being used for this calibration adjustment


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

Audubon = Audubon.dropna()
Adams = Adams.dropna()
Balboa = Balboa.dropna()
Browne = Browne.dropna()
Grant = Grant.dropna()
Jefferson = Jefferson.dropna()
Lidgerwood = Lidgerwood.dropna()
Regal = Regal.dropna()
Sheridan = Sheridan.dropna()
Stevens = Stevens.dropna()

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

#%%

n = 40
threshold = []
negative_count = []

while n < 90:
    
    # current threshold = 51
    # Apply calibrations (after hourly resample because the calibrations are based on hourly data)
    # high calibration
    Audubon['PM2_5_corrected'] = np.where((Audubon.PM2_5 > n), mlr_function_high_cal(mlr_high_audubon, Audubon), Audubon.PM2_5)  # high calibration adjustment
    # linear adjustement to reference node based on Paccar roof calibration
    Audubon['PM2_5_corrected'] = np.where((Audubon.PM2_5 < n), (Audubon.PM2_5-0.4207)/1.0739, Audubon.PM2_5_corrected)  # Paccar roof adjustment
    # apply mlr calibration from Augusta
    Audubon['PM2_5_corrected'] = np.where((Audubon.PM2_5 < n), Audubon.PM2_5*0.454-Audubon.Rel_humid*0.0483-Audubon.temp*0.0774+4.8242, Audubon.PM2_5_corrected)  # high calibration adjustment
    
    count = len(Audubon.loc[Audubon.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Audubon = pd.DataFrame()
df_Audubon['Threshold'] = threshold
df_Audubon['Negative_Count'] = negative_count

plot(df_Audubon, 'Audubon')

#%%
# current is 75

n = 40
threshold = []
negative_count = []

while n < 90:

    Adams['PM2_5_corrected'] = np.where((Adams.PM2_5 > n), mlr_function_high_cal(mlr_high_adams, Adams), Adams.PM2_5)  # high calibration adjustment
    Adams['PM2_5_corrected'] = np.where((Adams.PM2_5 < n), (Adams.PM2_5+0.93)/1.1554, Adams.PM2_5_corrected)  # Paccar roof adjustment
    Adams['PM2_5_corrected'] = np.where((Adams.PM2_5 < n), Adams.PM2_5*0.454-Adams.Rel_humid*0.0483-Adams.temp*0.0774+4.8242, Adams.PM2_5_corrected)  # high calibration adjustment

    count = len(Adams.loc[Adams.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Adams = pd.DataFrame()
df_Adams['Threshold'] = threshold
df_Adams['Negative_Count'] = negative_count

plot(df_Adams, 'Adams')
#%%

# current is 58

n = 40
threshold = []
negative_count = []

while n < 90:

    Balboa['PM2_5_corrected'] = np.where((Balboa.PM2_5 > n), mlr_function_high_cal(mlr_high_balboa, Balboa), Balboa.PM2_5)  # high calibration adjustment
    Balboa['PM2_5_corrected'] = np.where((Balboa.PM2_5 < n), (Balboa.PM2_5-0.2878)/1.2457, Balboa.PM2_5_corrected)  # Paccar roof adjustment
    Balboa['PM2_5_corrected'] = np.where((Balboa.PM2_5 < n), Balboa.PM2_5*0.454-Balboa.Rel_humid*0.0483-Balboa.temp*0.0774+4.8242, Balboa.PM2_5_corrected)  # high calibration adjustment

    count = len(Balboa.loc[Balboa.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Balboa = pd.DataFrame()
df_Balboa['Threshold'] = threshold
df_Balboa['Negative_Count'] = negative_count

plot(df_Balboa, 'Balboa')


#%%

# current is 74
n = 40
threshold = []
negative_count = []

while n < 90:
    
    Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 > n), mlr_function_high_cal(mlr_high_browne, Browne), Browne.PM2_5)  # high calibration adjustment
    Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 < n), (Browne.PM2_5-0.4771)/1.1082, Browne.PM2_5_corrected)  # Paccar roof adjustment
    Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 < n), Browne.PM2_5*0.454-Browne.Rel_humid*0.0483-Browne.temp*0.0774+4.8242, Browne.PM2_5_corrected)  # high calibration adjustment

    count = len(Browne.loc[Browne.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Browne = pd.DataFrame()
df_Browne['Threshold'] = threshold
df_Browne['Negative_Count'] = negative_count

plot(df_Browne, 'Browne')

#%%

# current is 77
n = 40
threshold = []
negative_count = []

while n < 90:

    Grant['PM2_5_corrected'] = np.where((Grant.PM2_5 > n), mlr_function_high_cal(mlr_high_grant, Grant), Grant.PM2_5)  # high calibration adjustment
    Grant['PM2_5_corrected'] = np.where((Grant.PM2_5 < n), (Grant.PM2_5+1.0965)/1.29, Grant.PM2_5_corrected)  # Paccar roof adjustment
    Grant['PM2_5_corrected'] = np.where((Grant.PM2_5 < n), Grant.PM2_5*0.454-Grant.Rel_humid*0.0483-Grant.temp*0.0774+4.8242, Grant.PM2_5_corrected)  # high calibration adjustment

    count = len(Grant.loc[Grant.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Grant = pd.DataFrame()
df_Grant['Threshold'] = threshold
df_Grant['Negative_Count'] = negative_count

plot(df_Grant, 'Grant')
#%%

# current is 73
n = 40
threshold = []
negative_count = []

while n < 90:

    Jefferson['PM2_5_corrected'] = np.where((Jefferson.PM2_5 > n), mlr_function_high_cal(mlr_high_jefferson, Jefferson), Jefferson.PM2_5)  # high calibration adjustment
    Jefferson['PM2_5_corrected'] = np.where((Jefferson.PM2_5 < n), (Jefferson.PM2_5+0.7099)/1.1458, Jefferson.PM2_5_corrected)  # Paccar roof adjustment
    Jefferson['PM2_5_corrected'] = np.where((Jefferson.PM2_5 < n), Jefferson.PM2_5*0.454-Jefferson.Rel_humid*0.0483-Jefferson.temp*0.0774+4.8242, Jefferson.PM2_5_corrected)  # high calibration adjustment

    count = len(Jefferson.loc[Jefferson.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Jefferson = pd.DataFrame()
df_Jefferson['Threshold'] = threshold
df_Jefferson['Negative_Count'] = negative_count

plot(df_Jefferson, 'Jefferson')
#%%

# current is 66
n = 40
threshold = []
negative_count = []

while n < 90:

    Lidgerwood['PM2_5_corrected'] = np.where((Lidgerwood.PM2_5 > n),  mlr_function_high_cal(mlr_high_lidgerwood, Lidgerwood), Lidgerwood.PM2_5)  # high calibration adjustment
    Lidgerwood['PM2_5_corrected'] = np.where((Lidgerwood.PM2_5 < n), (Lidgerwood.PM2_5-1.1306)/0.9566, Lidgerwood.PM2_5_corrected)  # Paccar roof adjustment
    Lidgerwood['PM2_5_corrected'] = np.where((Lidgerwood.PM2_5 < n), Lidgerwood.PM2_5*0.454-Lidgerwood.Rel_humid*0.0483-Lidgerwood.temp*0.0774+4.8242, Lidgerwood.PM2_5_corrected)  # high calibration adjustment

    count = len(Lidgerwood.loc[Lidgerwood.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Lidgerwood = pd.DataFrame()
df_Lidgerwood['Threshold'] = threshold
df_Lidgerwood['Negative_Count'] = negative_count

plot(df_Lidgerwood, 'Lidgerwood')
#%%

# current is 54
n = 40
threshold = []
negative_count = []

while n < 90:

    Regal['PM2_5_corrected'] = np.where((Regal.PM2_5 > n),  mlr_function_high_cal(mlr_high_regal, Regal), Regal.PM2_5)  # high calibration adjustment
    Regal['PM2_5_corrected'] = np.where((Regal.PM2_5 < n), (Regal.PM2_5-0.247)/0.9915, Regal.PM2_5_corrected)  # Paccar roof adjustment
    Regal['PM2_5_corrected'] = np.where((Regal.PM2_5 < n), Regal.PM2_5*0.454-Regal.Rel_humid*0.0483-Regal.temp*0.0774+4.8242, Regal.PM2_5_corrected)  # high calibration adjustment

    count = len(Regal.loc[Regal.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Regal = pd.DataFrame()
df_Regal['Threshold'] = threshold
df_Regal['Negative_Count'] = negative_count

plot(df_Regal, 'Regal')


#%%
# current is 82
n = 40
threshold = []
negative_count = []

while n < 90:

    Sheridan['PM2_5_corrected'] = np.where((Sheridan.PM2_5 > n), mlr_function_high_cal(mlr_high_sheridan, Sheridan), Sheridan.PM2_5)  # high calibration adjustment
    Sheridan['PM2_5_corrected'] = np.where((Sheridan.PM2_5 < n), (Sheridan.PM2_5+0.6958)/1.1468, Sheridan.PM2_5_corrected)  # Paccar roof adjustment
    Sheridan['PM2_5_corrected'] = np.where((Sheridan.PM2_5 < n), Sheridan.PM2_5*0.454-Sheridan.Rel_humid*0.0483-Sheridan.temp*0.0774+4.8242, Sheridan.PM2_5_corrected)  # high calibration adjustment

    count = len(Sheridan.loc[Sheridan.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Sheridan = pd.DataFrame()
df_Sheridan['Threshold'] = threshold
df_Sheridan['Negative_Count'] = negative_count

plot(df_Sheridan, 'Sheridan')

#%%
# current is 86
n = 40
threshold = []
negative_count = []

while n < 90:

    Stevens['PM2_5_corrected'] = np.where((Stevens.PM2_5 > n), mlr_function_high_cal(mlr_high_stevens, Stevens), Stevens.PM2_5)  # high calibration adjustment
    Stevens['PM2_5_corrected'] = np.where((Stevens.PM2_5 < n), (Stevens.PM2_5+0.8901)/1.2767, Stevens.PM2_5_corrected)  # Paccar roof adjustment
    Stevens['PM2_5_corrected'] = np.where((Stevens.PM2_5 < n), Stevens.PM2_5*0.454-Stevens.Rel_humid*0.0483-Stevens.temp*0.0774+4.8242, Stevens.PM2_5_corrected)  # high calibration adjustment

    count = len(Stevens.loc[Stevens.PM2_5_corrected < l])
    
    threshold.append(n)
    negative_count.append(count)
    
    n = n+1

df_Stevens = pd.DataFrame()
df_Stevens['Threshold'] = threshold
df_Stevens['Negative_Count'] = negative_count

plot(df_Stevens, 'Stevens')
