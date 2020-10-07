#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 19:11:54 2020

@author: matthew
"""

#%%
import pandas as pd
from glob import glob
import numpy as np
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from linear_plot_function import linear_plot
from high_cal_mlr_function import mlr_function_high_cal

PlotType = 'HTMLfile'

ModelType = 'mlr'    # options: rf, mlr, hybrid, linear
stdev_number = 1   # defines whether using 1 or 2 stdev for uncertainty

slope_sigma1 = 2       # percent uncertainty of SRCAA BAM calibration to reference clarity slope
slope_sigma2 = 4.5     # percent uncertainty of slope for paccar roof calibrations (square root of VAR slope from excel)
slope_sigma_paccar = 2     # percent uncertainty of slope for Paccar Clarity unit at SRCAA BAM calibration
sigma_i = 5            # uncertainty of Clarity measurements (arbitrary right now) in ug/m^3


# Choose dates of interest


# Date Range of interest
start_time = '2020-09-17 10:00'  
end_time = '2020-09-18 06:00'

interval = '60T'

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
Augusta = Augusta.resample(interval).mean()


#%%

ref_df = pd.DataFrame()

ref_df['Augusta'] = Augusta['PM2_5']
ref_df['Broadway'] = Broadway['PM2_5']
ref_df['Greenbluff'] = Greenbluff['PM2_5']
ref_df['stdev'] = ref_df.std(axis=1)
ref_df['ref_avg'] = ref_df.iloc[:, [0,1,2]].mean(axis=1)


calibration_df = pd.DataFrame()

calibration_df['Adams'] = Adams['PM2_5']
calibration_df['Adams_temp'] = Adams['temp']
calibration_df['Adams_rh'] = Adams['Rel_humid']

calibration_df['Audubon'] = Audubon['PM2_5']
calibration_df['Audubon_temp'] = Audubon['temp']
calibration_df['Audubon_rh'] = Audubon['Rel_humid']

calibration_df['Balboa'] = Balboa['PM2_5']
calibration_df['Balboa_temp'] = Balboa['temp']
calibration_df['Balboa_rh'] = Balboa['Rel_humid']

calibration_df['Browne'] = Browne['PM2_5']
calibration_df['Browne_temp'] = Browne['temp']
calibration_df['Browne_rh'] = Browne['Rel_humid']

calibration_df['Grant'] = Grant['PM2_5']
calibration_df['Grant_temp'] = Grant['temp']
calibration_df['Grant_rh'] = Grant['Rel_humid']

calibration_df['Jefferson'] = Jefferson['PM2_5']
calibration_df['Jefferson_temp'] = Jefferson['temp']
calibration_df['Jefferson_rh'] = Jefferson['Rel_humid']

calibration_df['Lidgerwood'] = Lidgerwood['PM2_5']
calibration_df['Lidgerwood_temp'] = Lidgerwood['temp']
calibration_df['Lidgerwood_rh'] = Lidgerwood['Rel_humid']

calibration_df['Regal'] = Regal['PM2_5']
calibration_df['Regal_temp'] = Regal['temp']
calibration_df['Regal_rh'] = Regal['Rel_humid']

calibration_df['Sheridan'] = Sheridan['PM2_5']
calibration_df['Sheridan_temp'] = Sheridan['temp']
calibration_df['Sheridan_rh'] = Sheridan['Rel_humid']

calibration_df['Stevens'] = Stevens['PM2_5']
calibration_df['Stevens_temp'] = Stevens['temp']
calibration_df['Stevens_rh'] = Stevens['Rel_humid']

calibration_df['ref_avg'] = ref_df['ref_avg']
calibration_df['ref_stdev'] = ref_df['stdev']
calibration_df['time'] = calibration_df.index

#%%

#date_range = '9_13_00_00_to_9_14_06_00'
#date_range = '9_15_16_00_to_9_16_12_00'
date_range = '9_17_10_00_to_9_18_06_00'

calibration_df.to_csv('/Users/matthew/Desktop/data/high_calibration/high_calibration' + '_' + date_range + '.csv', index=False)


#%%
#Import high calibration data set

calibration_df = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/high_calibration/high_calibration*.csv')
files.sort()
for file in files:
    calibration_df = pd.concat([calibration_df, pd.read_csv(file)], sort=False)


calibration_df['time'] = pd.to_datetime(calibration_df['time'])
calibration_df = calibration_df.sort_values('time')
calibration_df.index = calibration_df.time
calibration_df = calibration_df.dropna()
#%%
# linear adjustments based on equation from linear regression (found from running the plot linear regression functionbelow) 
# between each Clarity node and the avg. ref measurements

calibration_df['Adams_linear_cal'] = (calibration_df['Adams']-36.45)/1.03
calibration_df['Audubon_linear_cal'] = (calibration_df['Audubon']-36.06)/0.55
calibration_df['Balboa_linear_cal'] = (calibration_df['Balboa']-20.74)/1.41
calibration_df['Browne_linear_cal'] = (calibration_df['Browne']-42.09)/0.85
calibration_df['Grant_linear_cal'] = (calibration_df['Grant']-37.25)/1.16
calibration_df['Jefferson_linear_cal'] = (calibration_df['Jefferson']-37.85)/0.96
calibration_df['Lidgerwood_linear_cal'] = (calibration_df['Lidgerwood']-29.33)/0.95
calibration_df['Regal_linear_cal'] = (calibration_df['Regal']-29.86)/0.78
calibration_df['Sheridan_linear_cal'] = (calibration_df['Sheridan']-42.46)/0.97
calibration_df['Stevens_linear_cal'] = (calibration_df['Stevens']-39.23)/1.2

#%%

# mlr calibration

calibration_df['Adams_mlr'] = mlr_function_high_cal(calibration_df, 'Adams')
calibration_df['Audubon_mlr'] = mlr_function_high_cal(calibration_df, 'Audubon')
calibration_df['Balboa_mlr'] = mlr_function_high_cal(calibration_df, 'Balboa')
calibration_df['Browne_mlr'] = mlr_function_high_cal(calibration_df, 'Browne')
calibration_df['Grant_mlr'] = mlr_function_high_cal(calibration_df, 'Grant')
calibration_df['Jefferson_mlr'] = mlr_function_high_cal(calibration_df, 'Jefferson')
calibration_df['Lidgerwood_mlr'] = mlr_function_high_cal(calibration_df, 'Lidgerwood')
calibration_df['Regal_mlr'] = mlr_function_high_cal(calibration_df, 'Regal')
calibration_df['Sheridan_mlr'] = mlr_function_high_cal(calibration_df, 'Sheridan')
calibration_df['Stevens_mlr'] = mlr_function_high_cal(calibration_df, 'Stevens')

#%%

# Plot all Clarity PM 2.5 using interactive legend (mute)

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/high_calibration/raw_PM2_5_measurements.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')

p1.title.text = 'Smoke Event Raw PM 2.5'

p1.line(calibration_df.index,     calibration_df.Audubon,     legend='Audubon',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
p1.line(calibration_df.index,       calibration_df.Adams,       legend='Adams',         color='blue',        line_width=2, muted_color='blue', muted_alpha=0.2)
p1.line(calibration_df.index,      calibration_df.Balboa,      legend='Balboa',        color='red',         line_width=2, muted_color='red', muted_alpha=0.2)
p1.line(calibration_df.index,      calibration_df.Browne,      legend='Browne',        color='black',       line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(calibration_df.index,       calibration_df.Grant,       legend='Grant',         color='purple',      line_width=2, muted_color='purple', muted_alpha=0.2)
p1.line(calibration_df.index,   calibration_df.Jefferson,   legend='Jefferson',     color='brown',       line_width=2, muted_color='brown', muted_alpha=0.2)
p1.line(calibration_df.index,  calibration_df.Lidgerwood,  legend='Lidgerwood',    color='orange',      line_width=2, muted_color='orange', muted_alpha=0.2)
p1.line(calibration_df.index,       calibration_df.Regal,       legend='Regal',         color='khaki',       line_width=2, muted_color='khaki', muted_alpha=0.2)
p1.line(calibration_df.index,    calibration_df.Sheridan,    legend='Sheridan',      color='deepskyblue', line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
p1.line(calibration_df.index,     calibration_df.Stevens,     legend='Stevens',       color='grey',        line_width=2, muted_color='grey', muted_alpha=0.2)
p1.line(calibration_df.index,     calibration_df.ref_avg,     legend='Ref Avg',       color='gold',        line_width=2, muted_color='gold', muted_alpha=0.2)


p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Smoke Event Raw PM 2.5")

tabs = Tabs(tabs=[ tab1])

show(tabs)

#%%

linear_plot(calibration_df.ref_avg, calibration_df.Audubon,'Audubon')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Adams,'Adams')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Balboa,'Balboa')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Browne,'Browne')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Grant,'Grant')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Jefferson,'Jefferson')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Lidgerwood,'Lidgerwood')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Regal,'Regal')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Sheridan,'Sheridan')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Stevens,'Stevens')


#%%

# Plot linearly corrected Clarity values for high calibration time period
linear_plot(calibration_df.ref_avg, calibration_df.Audubon_linear_cal,'Audubon')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Adams_linear_cal,'Adams')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Balboa_linear_cal,'Balboa')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Browne_linear_cal,'Browne')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Grant_linear_cal,'Grant')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Jefferson_linear_cal,'Jefferson')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Lidgerwood_linear_cal,'Lidgerwood')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Regal_linear_cal,'Regal')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Sheridan_linear_cal,'Sheridan')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Stevens_linear_cal,'Stevens')

#%%

# Plot mlr correccted Clarity values for high calibration time period
linear_plot(calibration_df.ref_avg, calibration_df.Audubon_mlr,'Audubon mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Adams_mlr,'Adams mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Balboa_mlr,'Balboa mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Browne_mlr,'Browne mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Grant_mlr,'Grant mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Jefferson_mlr,'Jefferson mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Lidgerwood_mlr,'Lidgerwood mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Regal_mlr,'Regal mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Sheridan_mlr,'Sheridan mlr')
#%%
linear_plot(calibration_df.ref_avg, calibration_df.Stevens_mlr,'Stevens mlr')













