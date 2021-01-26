#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 12:11:03 2020

@author: matthew
"""

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
from high_cal_mlr_function_generator import high_cal_setup, generate_mlr_function_high_cal
from mlr_function_for_combined_data import mlr_function_general
# Import curve fitting package from scipy
from scipy.optimize import curve_fit


# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a*np.power(x, b)
def power_law_cal(y, a , b):
    return (y/a)**(1/b)

PlotType = 'HTMLfile'

ModelType = 'mlr'    # options: rf, mlr, hybrid, linear
stdev_number = 1   # defines whether using 1 or 2 stdev for uncertainty

slope_sigma1 = 2       # percent uncertainty of SRCAA BAM calibration to reference clarity slope
slope_sigma2 = 4.5     # percent uncertainty of slope for paccar roof calibrations (square root of VAR slope from excel)
slope_sigma_paccar = 2     # percent uncertainty of slope for Paccar Clarity unit at SRCAA BAM calibration
sigma_i = 5            # uncertainty of Clarity measurements (arbitrary right now) in ug/m^3


# Choose dates of interest


# Choose dates of interest
#Augusta Times
start_time = '2019-12-17 00:00'
end_time = '2020-03-05 00:00'

interval = '60T'


#%%


Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv')
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)

Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]

#Read in SRCAA Augusta site BAM data

Augusta_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)
Augusta_All['PM2_5_corrected'] = Augusta_All['PM2_5']    # creates column with same values so loops work below
Augusta_All['Location'] = 'Augusta'


Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time] 


Reference = Reference.resample(interval).mean() 
Augusta = Augusta.resample(interval).mean()


#%%

calibration_df = pd.DataFrame()

# create df with applying the reverse Paccar roof calibration to the reference node for each of the Clarity nodes to approximate 
# what their raw data readings would have been (just for the winter SRCAA overlap data) - I double checked and the below equations were
# only applied to the winter data not the high cal data

#Adams_All['PM2_5_corrected'] = (Adams_All['PM2_5']+0.93)/1.1554
calibration_df['Adams'] = Reference['PM2_5']*1.1554-0.93
calibration_df['Adams_temp'] = Reference['temp']
calibration_df['Adams_rh'] = Reference['Rel_humid']


#Audubon_All['PM2_5_corrected'] = (Audubon_All['PM2_5']-0.42073)/1.0739
calibration_df['Audubon'] = Reference['PM2_5']*1.0739+0.42073
calibration_df['Audubon_temp'] = Reference['temp']
calibration_df['Audubon_rh'] = Reference['Rel_humid']


#Balboa_All['PM2_5_corrected'] = (Balboa_All['PM2_5']-0.2878)/1.2457 
calibration_df['Balboa'] = Reference['PM2_5']*1.2457+0.2878
calibration_df['Balboa_temp'] = Reference['temp']
calibration_df['Balboa_rh'] = Reference['Rel_humid']


#Browne_All['PM2_5_corrected'] = (Browne_All['PM2_5']-0.4771)/1.1082 
calibration_df['Browne'] = Reference['PM2_5']*1.1082+0.4771
calibration_df['Browne_temp'] = Reference['temp']
calibration_df['Browne_rh'] = Reference['Rel_humid']


#Grant_All['PM2_5_corrected'] = (Grant_All['PM2_5']+1.0965)/1.29
calibration_df['Grant'] = Reference['PM2_5']*1.29-1.0965
calibration_df['Grant_temp'] = Reference['temp']
calibration_df['Grant_rh'] = Reference['Rel_humid']


#Jefferson_All['PM2_5_corrected'] = (Jefferson_All['PM2_5']+0.7099)/1.1458
calibration_df['Jefferson'] = Reference['PM2_5']*1.1458-0.7099
calibration_df['Jefferson_temp'] = Reference['temp']
calibration_df['Jefferson_rh'] = Reference['Rel_humid']


#Lidgerwood_All['PM2_5_corrected'] = (Lidgerwood_All['PM2_5']-1.1306)/0.9566  
calibration_df['Lidgerwood'] = Reference['PM2_5']*0.9566+1.1306
calibration_df['Lidgerwood_temp'] = Reference['temp']
calibration_df['Lidgerwood_rh'] = Reference['Rel_humid']


#Regal_All['PM2_5_corrected'] = (Regal_All['PM2_5']-0.247)/0.9915    
calibration_df['Regal'] = Reference['PM2_5']*0.9915+0.247
calibration_df['Regal_temp'] = Reference['temp']
calibration_df['Regal_rh'] = Reference['Rel_humid']


#Sheridan_All['PM2_5_corrected'] = (Sheridan_All['PM2_5']+0.6958)/1.1468 
calibration_df['Sheridan'] = Reference['PM2_5']*1.1468-0.6958
calibration_df['Sheridan_temp'] = Reference['temp']
calibration_df['Sheridan_rh'] = Reference['Rel_humid']


#Stevens_All['PM2_5_corrected'] = (Stevens_All['PM2_5']+0.8901)/1.2767
calibration_df['Stevens'] = Reference['PM2_5']*1.2767-0.8901
calibration_df['Stevens_temp'] = Reference['temp']
calibration_df['Stevens_rh'] = Reference['Rel_humid']


# Note that the names ref_avg just refers to the Augusta BAM (not an average of anything)
# it is just named like this so that the data from the high calibration can be loaded in 
# with the same column headers as that data is compared to the average of the reference instruments in Spokane
calibration_df['ref_avg'] = Augusta['PM2_5']




calibration_df['time'] = calibration_df.index

#%%

date_range = '12_17__19_17_00_to_03_05_20_00_00'

calibration_df.to_csv('/Users/matthew/Desktop/data/combined_calibration/combined_calibration' + '_' + date_range + '.csv', index=False)


#%%
#Import high calibration data set

calibration_df = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/combined_calibration/*.csv')
files.sort()
for file in files:
    calibration_df = pd.concat([calibration_df, pd.read_csv(file)], sort=False)
    
high_calibration_df = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/combined_calibration/high*.csv')
files.sort()
for file in files:
    high_calibration_df = pd.concat([high_calibration_df, pd.read_csv(file)], sort=False)
    
winter_calibration_df = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/combined_calibration/winter*.csv')
files.sort()
for file in files:
    winter_calibration_df = pd.concat([winter_calibration_df, pd.read_csv(file)], sort=False)
#%%

calibration_df['time'] = pd.to_datetime(calibration_df['time'])
calibration_df = calibration_df.sort_values('time')
calibration_df.index = calibration_df.time
del calibration_df['ref_stdev']
calibration_df = calibration_df.dropna()

high_calibration_df['time'] = pd.to_datetime(high_calibration_df['time'])
high_calibration_df = high_calibration_df.sort_values('time')
high_calibration_df.index = high_calibration_df.time
del high_calibration_df['ref_stdev']
high_calibration_df = high_calibration_df.dropna()

winter_calibration_df['time'] = pd.to_datetime(winter_calibration_df['time'])
winter_calibration_df = winter_calibration_df.sort_values('time')
winter_calibration_df.index = winter_calibration_df.time
winter_calibration_df = winter_calibration_df.dropna()

#%%
# generate the mlr functions for calibrating both high and winter calibration

mlr_high_audubon = generate_mlr_function_high_cal(high_calibration_df, 'Audubon')
mlr_high_adams = generate_mlr_function_high_cal(high_calibration_df, 'Adams')
mlr_high_balboa = generate_mlr_function_high_cal(high_calibration_df, 'Balboa')
mlr_high_browne = generate_mlr_function_high_cal(high_calibration_df, 'Browne')
mlr_high_grant = generate_mlr_function_high_cal(high_calibration_df, 'Grant')
mlr_high_jefferson = generate_mlr_function_high_cal(high_calibration_df, 'Jefferson')
mlr_high_lidgerwood = generate_mlr_function_high_cal(high_calibration_df, 'Lidgerwood')
mlr_high_regal = generate_mlr_function_high_cal(high_calibration_df, 'Regal')
mlr_high_sheridan = generate_mlr_function_high_cal(high_calibration_df, 'Sheridan')
mlr_high_stevens = generate_mlr_function_high_cal(high_calibration_df, 'Stevens')

mlr_winter_audubon = generate_mlr_function_high_cal(winter_calibration_df, 'Audubon')
mlr_winter_adams = generate_mlr_function_high_cal(winter_calibration_df, 'Adams')
mlr_winter_balboa = generate_mlr_function_high_cal(winter_calibration_df, 'Balboa')
mlr_winter_browne = generate_mlr_function_high_cal(winter_calibration_df, 'Browne')
mlr_winter_grant = generate_mlr_function_high_cal(winter_calibration_df, 'Grant')
mlr_winter_jefferson = generate_mlr_function_high_cal(winter_calibration_df, 'Jefferson')
mlr_winter_lidgerwood = generate_mlr_function_high_cal(winter_calibration_df, 'Lidgerwood')
mlr_winter_regal = generate_mlr_function_high_cal(winter_calibration_df, 'Regal')
mlr_winter_sheridan = generate_mlr_function_high_cal(winter_calibration_df, 'Sheridan')
mlr_winter_stevens = generate_mlr_function_high_cal(winter_calibration_df, 'Stevens')

mlr_combined_audubon = generate_mlr_function_high_cal(calibration_df, 'Audubon')
mlr_combined_adams = generate_mlr_function_high_cal(calibration_df, 'Adams')
mlr_combined_balboa = generate_mlr_function_high_cal(calibration_df, 'Balboa')
mlr_combined_browne = generate_mlr_function_high_cal(calibration_df, 'Browne')
mlr_combined_grant = generate_mlr_function_high_cal(calibration_df, 'Grant')
mlr_combined_jefferson = generate_mlr_function_high_cal(calibration_df, 'Jefferson')
mlr_combined_lidgerwood = generate_mlr_function_high_cal(calibration_df, 'Lidgerwood')
mlr_combined_regal = generate_mlr_function_high_cal(calibration_df, 'Regal')
mlr_combined_sheridan = generate_mlr_function_high_cal(calibration_df, 'Sheridan')
mlr_combined_stevens = generate_mlr_function_high_cal(calibration_df, 'Stevens')

#%%
# linear adjustments based on equation from linear regression (found from running the plot linear regression functionbelow) 
# between each Clarity node and the avg. ref measurements

#calibration_df['Adams_linear_cal'] = (calibration_df['Adams']-36.45)/1.03
#calibration_df['Audubon_linear_cal'] = (calibration_df['Audubon']-36.06)/0.55
#calibration_df['Balboa_linear_cal'] = (calibration_df['Balboa']-20.74)/1.41
#calibration_df['Browne_linear_cal'] = (calibration_df['Browne']-42.09)/0.85
#calibration_df['Grant_linear_cal'] = (calibration_df['Grant']-37.25)/1.16
#calibration_df['Jefferson_linear_cal'] = (calibration_df['Jefferson']-37.85)/0.96
#calibration_df['Lidgerwood_linear_cal'] = (calibration_df['Lidgerwood']-29.33)/0.95
#calibration_df['Regal_linear_cal'] = (calibration_df['Regal']-29.86)/0.78
#calibration_df['Sheridan_linear_cal'] = (calibration_df['Sheridan']-42.46)/0.97
#calibration_df['Stevens_linear_cal'] = (calibration_df['Stevens']-39.23)/1.2

#%%

# mlr calibration

#calibration_df['Adams_mlr'] = mlr_function_general(calibration_df, 'Adams')
#calibration_df['Audubon_mlr'] = mlr_function_general(calibration_df, 'Audubon')
#calibration_df['Balboa_mlr'] = mlr_function_general(calibration_df, 'Balboa')
#calibration_df['Browne_mlr'] = mlr_function_general(calibration_df, 'Browne')
#calibration_df['Grant_mlr'] = mlr_function_general(calibration_df, 'Grant')
#calibration_df['Jefferson_mlr'] = mlr_function_general(calibration_df, 'Jefferson')
#calibration_df['Lidgerwood_mlr'] = mlr_function_general(calibration_df, 'Lidgerwood')
#calibration_df['Regal_mlr'] = mlr_function_general(calibration_df, 'Regal')
#calibration_df['Sheridan_mlr'] = mlr_function_general(calibration_df, 'Sheridan')
#calibration_df['Stevens_mlr'] = mlr_function_general(calibration_df, 'Stevens')

high_calibration_df['Adams_mlr_high'] = mlr_function_general(mlr_high_adams, high_calibration_df, 'Adams')
high_calibration_df['Audubon_mlr_high'] = mlr_function_general(mlr_high_audubon, high_calibration_df, 'Audubon')
high_calibration_df['Balboa_mlr_high'] = mlr_function_general(mlr_high_balboa, high_calibration_df, 'Balboa')
high_calibration_df['Browne_mlr_high'] = mlr_function_general(mlr_high_browne, high_calibration_df, 'Browne')
high_calibration_df['Grant_mlr_high'] = mlr_function_general(mlr_high_grant, high_calibration_df,'Grant')
high_calibration_df['Jefferson_mlr_high'] = mlr_function_general(mlr_high_jefferson, high_calibration_df,'Jefferson')
high_calibration_df['Lidgerwood_mlr_high'] = mlr_function_general(mlr_high_lidgerwood, high_calibration_df,'Lidgerwood')
high_calibration_df['Regal_mlr_high'] = mlr_function_general(mlr_high_regal, high_calibration_df,'Regal')
high_calibration_df['Sheridan_mlr_high'] = mlr_function_general(mlr_high_sheridan, high_calibration_df,'Sheridan')
high_calibration_df['Stevens_mlr_high'] = mlr_function_general(mlr_high_stevens, high_calibration_df,'Stevens')


winter_calibration_df['Adams_mlr_winter'] = mlr_function_general(mlr_winter_adams, winter_calibration_df, 'Adams')
winter_calibration_df['Audubon_mlr_winter'] = mlr_function_general(mlr_winter_audubon, winter_calibration_df, 'Audubon')
winter_calibration_df['Balboa_mlr_winter'] = mlr_function_general(mlr_winter_balboa, winter_calibration_df,'Balboa')
winter_calibration_df['Browne_mlr_winter'] = mlr_function_general(mlr_winter_browne, winter_calibration_df,'Browne')
winter_calibration_df['Grant_mlr_winter'] = mlr_function_general(mlr_winter_grant, winter_calibration_df,'Grant')
winter_calibration_df['Jefferson_mlr_winter'] = mlr_function_general(mlr_winter_jefferson, winter_calibration_df,'Jefferson')
winter_calibration_df['Lidgerwood_mlr_winter'] = mlr_function_general(mlr_winter_lidgerwood, winter_calibration_df,'Lidgerwood')
winter_calibration_df['Regal_mlr_winter'] = mlr_function_general(mlr_winter_regal, winter_calibration_df,'Regal')
winter_calibration_df['Sheridan_mlr_winter'] = mlr_function_general(mlr_winter_sheridan, winter_calibration_df,'Sheridan')
winter_calibration_df['Stevens_mlr_winter'] = mlr_function_general(mlr_winter_stevens, winter_calibration_df,'Stevens')

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

# plot combined calibration data
#p1.line(calibration_df.index,     calibration_df.Audubon,     legend='Audubon',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
#p1.line(calibration_df.index,       calibration_df.Adams,       legend='Adams',         color='blue',        line_width=2, muted_color='blue', muted_alpha=0.2)
#p1.line(calibration_df.index,      calibration_df.Balboa,      legend='Balboa',        color='red',         line_width=2, muted_color='red', muted_alpha=0.2)
#p1.line(calibration_df.index,      calibration_df.Browne,      legend='Browne',        color='black',       line_width=2, muted_color='black', muted_alpha=0.2)
#p1.line(calibration_df.index,       calibration_df.Grant,       legend='Grant',         color='purple',      line_width=2, muted_color='purple', muted_alpha=0.2)
#p1.line(calibration_df.index,   calibration_df.Jefferson,   legend='Jefferson',     color='brown',       line_width=2, muted_color='brown', muted_alpha=0.2)
#p1.line(calibration_df.index,  calibration_df.Lidgerwood,  legend='Lidgerwood',    color='orange',      line_width=2, muted_color='orange', muted_alpha=0.2)
#p1.line(calibration_df.index,       calibration_df.Regal,       legend='Regal',         color='khaki',       line_width=2, muted_color='khaki', muted_alpha=0.2)
#p1.line(calibration_df.index,    calibration_df.Sheridan,    legend='Sheridan',      color='deepskyblue', line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
#p1.line(calibration_df.index,     calibration_df.Stevens,     legend='Stevens',       color='grey',        line_width=2, muted_color='grey', muted_alpha=0.2)
#p1.line(calibration_df.index,     calibration_df.ref_avg,     legend='Augusta BAM',       color='gold',        line_width=2, muted_color='gold', muted_alpha=0.2)


# plot high calibration data
p1.line(high_calibration_df.index,     high_calibration_df.Audubon,     legend='Audubon',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
p1.line(high_calibration_df.index,       high_calibration_df.Adams,       legend='Adams',         color='blue',        line_width=2, muted_color='blue', muted_alpha=0.2)
p1.line(high_calibration_df.index,      high_calibration_df.Balboa,      legend='Balboa',        color='red',         line_width=2, muted_color='red', muted_alpha=0.2)
p1.line(high_calibration_df.index,      high_calibration_df.Browne,      legend='Browne',        color='black',       line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(high_calibration_df.index,       high_calibration_df.Grant,       legend='Grant',         color='purple',      line_width=2, muted_color='purple', muted_alpha=0.2)
p1.line(high_calibration_df.index,   high_calibration_df.Jefferson,   legend='Jefferson',     color='brown',       line_width=2, muted_color='brown', muted_alpha=0.2)
p1.line(high_calibration_df.index,  high_calibration_df.Lidgerwood,  legend='Lidgerwood',    color='orange',      line_width=2, muted_color='orange', muted_alpha=0.2)
p1.line(high_calibration_df.index,       high_calibration_df.Regal,       legend='Regal',         color='khaki',       line_width=2, muted_color='khaki', muted_alpha=0.2)
p1.line(high_calibration_df.index,    high_calibration_df.Sheridan,    legend='Sheridan',      color='deepskyblue', line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
p1.line(high_calibration_df.index,     high_calibration_df.Stevens,     legend='Stevens',       color='grey',        line_width=2, muted_color='grey', muted_alpha=0.2)
p1.line(high_calibration_df.index,     high_calibration_df.ref_avg,     legend='Augusta BAM',       color='gold',        line_width=2, muted_color='gold', muted_alpha=0.2)


# plot winter calibration data

p1.line(winter_calibration_df.index,     winter_calibration_df.Audubon,     legend='Audubon',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
p1.line(winter_calibration_df.index,       winter_calibration_df.Adams,       legend='Adams',         color='blue',        line_width=2, muted_color='blue', muted_alpha=0.2)
p1.line(winter_calibration_df.index,      winter_calibration_df.Balboa,      legend='Balboa',        color='red',         line_width=2, muted_color='red', muted_alpha=0.2)
p1.line(winter_calibration_df.index,      winter_calibration_df.Browne,      legend='Browne',        color='black',       line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(winter_calibration_df.index,       winter_calibration_df.Grant,       legend='Grant',         color='purple',      line_width=2, muted_color='purple', muted_alpha=0.2)
p1.line(winter_calibration_df.index,   winter_calibration_df.Jefferson,   legend='Jefferson',     color='brown',       line_width=2, muted_color='brown', muted_alpha=0.2)
p1.line(winter_calibration_df.index,  winter_calibration_df.Lidgerwood,  legend='Lidgerwood',    color='orange',      line_width=2, muted_color='orange', muted_alpha=0.2)
p1.line(winter_calibration_df.index,       winter_calibration_df.Regal,       legend='Regal',         color='khaki',       line_width=2, muted_color='khaki', muted_alpha=0.2)
p1.line(winter_calibration_df.index,    winter_calibration_df.Sheridan,    legend='Sheridan',      color='deepskyblue', line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
p1.line(winter_calibration_df.index,     winter_calibration_df.Stevens,     legend='Stevens',       color='grey',        line_width=2, muted_color='grey', muted_alpha=0.2)
p1.line(winter_calibration_df.index,     winter_calibration_df.ref_avg,     legend='Augusta BAM',       color='gold',        line_width=2, muted_color='gold', muted_alpha=0.2)


p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Combined Cal Raw Data")

# Plot scatter of combined Browne data and fit power function (test to see if works and then turn into function call for other locations)

p2 = figure(plot_width=900,
            plot_height=450,
            #x_axis_type='datetime',
            x_axis_label='Reference (ug/m3)',
            y_axis_label='Browne (ug/m3)')

p2.title.text = 'Browne Combined PM 2.5'

Browne_df = pd.DataFrame()
Browne_df['ref_avg'] = calibration_df.ref_avg
Browne_df['Browne'] = calibration_df.Browne
Browne_df = Browne_df[Browne_df['ref_avg'] > 0]
Browne_df.to_csv('/Users/matthew/Desktop/Browne_power_cal.csv', index=False)

xdata = Browne_df[['ref_avg']].to_numpy()
xdata = xdata[:, 0]
print(type(xdata))
ydata = Browne_df[['Browne']].to_numpy()
ydata = ydata[:, 0]


# Fit the dummy power-law data
pars, cov = curve_fit(f=power_law, xdata=xdata, ydata=ydata, p0=[0, 0], bounds=(-np.inf, np.inf))
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))
# Calculate the power calibrated data
Browne_df['Browne_power_fit'] = power_law_cal(ydata, *pars)


# plot combined calibration data
p2.scatter(xdata,     ydata,     legend='Browne',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
p2.scatter(xdata, power_law(xdata, *pars), legend='Power fit',       color='black',       line_width=2, muted_color='green', muted_alpha=0.2)
tab2 = Panel(child=p2, title="Combined Browne Raw Data")


p3 = figure(plot_width=900,
            plot_height=450,
            #x_axis_type='datetime',
            x_axis_label='Reference (ug/m3)',
            y_axis_label='Calibrated PM 2.5 (ug/m3)')

p3.title.text = 'Browne Combined Power Cal PM 2.5'

# plot combined calibration data
p3.scatter(Browne_df.ref_avg,    Browne_df.Browne_power_fit,     legend='Browne',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
p3.scatter(Browne_df.ref_avg,    Browne_df.ref_avg,     legend='1 to 1 line',       color='red',       line_width=2, muted_color='green', muted_alpha=0.2)

tab3 = Panel(child=p3, title="Combined Browne Calibrated Data")

tabs = Tabs(tabs=[ tab1, tab2, tab3])

show(tabs)


#%%
# if want to plot a single linear regression through the combined data, need to change the inputs to the calibration df and don't
# add in the second set of data (ie normally have high cal and winter cal df's input into the function)


linear_plot(high_calibration_df.ref_avg, high_calibration_df.Audubon,winter_calibration_df.ref_avg, winter_calibration_df.Audubon,'Audubon',2)

#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Adams,winter_calibration_df.ref_avg, winter_calibration_df.Adams,'Adams',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Balboa,winter_calibration_df.ref_avg, winter_calibration_df.Balboa,'Balboa',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Browne,winter_calibration_df.ref_avg, winter_calibration_df.Browne,'Browne',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Grant,winter_calibration_df.ref_avg, winter_calibration_df.Grant,'Grant',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Jefferson,winter_calibration_df.ref_avg, winter_calibration_df.Jefferson,'Jefferson',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Lidgerwood,winter_calibration_df.ref_avg, winter_calibration_df.Lidgerwood,'Lidgerwood',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Regal,winter_calibration_df.ref_avg, winter_calibration_df.Regal,'Regal',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Sheridan,winter_calibration_df.ref_avg, winter_calibration_df.Sheridan,'Sheridan',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Stevens,winter_calibration_df.ref_avg, winter_calibration_df.Stevens,'Stevens',2)
#%%

# Plot mlr calibrated data at the same time and get linear regression equations for corrected data
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Audubon_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Audubon_mlr_winter,'Audubon',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Adams_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Adams_mlr_winter,'Adams',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Balboa_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Balboa_mlr_winter,'Balboa',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Browne_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Browne_mlr_winter,'Browne',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Grant_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Grant_mlr_winter,'Grant',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Jefferson_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Jefferson_mlr_winter,'Jefferson',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Lidgerwood_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Lidgerwood_mlr_winter,'Lidgerwood',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Regal_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Regal_mlr_winter,'Regal',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Sheridan_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Sheridan_mlr_winter,'Sheridan',2)
#%%
linear_plot(high_calibration_df.ref_avg, high_calibration_df.Stevens_mlr_high,winter_calibration_df.ref_avg, winter_calibration_df.Stevens_mlr_winter,'Stevens',2)

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

import shapely
from shapely.geometry import LineString, Point

# Given these endpoints
#line 1
A = (-5, 1)
B = (7, 5)

#line 2
C = (0, 0)
D = (8, 8)

print(A)
print(B)




line1 = LineString([A, B])
print(type(line1))
line2 = LineString([C, D])

int_pt = line1.intersection(line2)
print(type(int_pt))
point_of_intersection = int_pt.x, int_pt.y

print(point_of_intersection)

#%%

# Generate dummy dataset
x_dummy = np.linspace(start=1, stop=1000, num=100)
print(type(x_dummy))
y_dummy = power_law(x_dummy, 1, 0.5)
# Add noise from a Gaussian distribution
noise = 1.5*np.random.normal(size=y_dummy.size)
y_dummy = y_dummy + noise



# Fit the dummy power-law data
pars, cov = curve_fit(f=power_law, xdata=x_dummy, ydata=y_dummy, p0=[0, 0], bounds=(-np.inf, np.inf))
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))
# Calculate the residuals
res = y_dummy - power_law(x_dummy, *pars)


#%%



xdata = calibration_df[['ref_avg']].to_numpy()
xdata = xdata[:, 0]
print(type(xdata))
ydata = calibration_df[['Browne']].to_numpy()
ydata = ydata[:, 0]


# Fit the dummy power-law data
pars, cov = curve_fit(f=power_law, xdata=xdata, ydata=ydata, p0=[0, 0], bounds=(-np.inf, np.inf))
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))
# Calculate the residuals
calibration_df['Browne_power_fit'] = power_law(xdata, *pars)
