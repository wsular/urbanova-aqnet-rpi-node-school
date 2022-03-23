#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:34:21 2020

@author: matthew
"""
import pandas as pd
from glob import glob
import matplotlib as plt
import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
from bokeh.io import export_png
from bokeh.layouts import gridplot
import metpy
from scipy import stats
from scipy import optimize  
import statsmodels.api as sm
import statistics
from gaussian_fit_function import gaussian_fit
from bokeh.models import HoverTool
from spec_humid import spec_humid
from load_indoor_data import load_indoor
from limit_of_detection import lod
#import copy
#from random_forest_function_test import rf, evaluate_model
#from mlr_function import mlr_function, mlr_model
from Augusta_hybrid_calibration import hybrid_function
from linear_plot_function import linear_plot
#%%

Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv')
#%%
for name in glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv'):
    print(name)
#%%
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)
    
Paccar_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Paccar*.csv')
files.sort()
for file in files:
    Paccar_All = pd.concat([Paccar_All, pd.read_csv(file)], sort=False)


#Read in SRCAA Augusta site BAM data

Augusta_All = pd.DataFrame({})

# Use BAM data from AIRNOW
#files = glob('/Users/matthew/Desktop/data/AirNow/Augusta_AirNow_updated.csv')


# Use BAM data from SRCAA
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)



#%%
    
threshold = 3.48         # threshold for splitting df's for "blank" measurements of BAM for calculating LOD

# Choose dates of interest
    # Augusta overlap period
start_time = '2019-12-17 15:00'
end_time = '2020-03-05 23:00'

# dates that Clarity used
# whole range
#start_time = '2019-12-18 00:00'
#end_time = '2020-03-05 23:00'

# Clarity 'Jan'
#start_time = '2019-12-18 00:00'
#end_time = '2020-01-31 23:00'

# Clarity Feb
#start_time = '2020-02-01 00:00'
#end_time = '2020-02-29 23:00'

# March 1-5
#start_time = '2020-03-01 00:00'
#end_time = '2020-03-05 23:00'

interval = '60T'
#interval = '15T'
#interval = '24H'
#%%

#Compare Clarity Units to Augusta SRCAA BAM as time series
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure


Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time]
Augusta = Augusta.resample(interval).mean()
Augusta['time'] = Augusta.index
#Augusta = Augusta.interpolate(method='linear')


df1 = Augusta[Augusta.PM2_5 >= 25]


Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar = Paccar_All.loc[start_time:end_time]
Paccar = Paccar.resample(interval).mean()

Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]
Reference = Reference.resample(interval).mean()  
                   
#%%
# hybrid model performance metrics
Reference_hybrid_calibrated = hybrid_function(rf, mlr_model, Reference, 'Reference') 
Reference_hybrid_calibrated = Reference_hybrid_calibrated.sort_index()

residuals = Augusta.PM2_5 - Reference_hybrid_calibrated.PM2_5_corrected
print('calibrated residuals stdev = ', residuals.std())  
Reference_hybrid_calibrated['Augusta'] = Augusta.PM2_5
#%%
linear_plot(Reference_hybrid_calibrated.Augusta, Reference_hybrid_calibrated.PM2_5_corrected, 
            Reference_hybrid_calibrated.Augusta, Reference_hybrid_calibrated.PM2_5_corrected,'Clarity Reference', 1)   # used for plotting the corrected data with 1 equation for each region
#linear_plot(audubon.ref_value, audubon.indoor_corrected, audubon.ref_value, audubon.indoor,'audubon', 1, residuals_check = 1, residuals = audubon.prediction_residuals)
#%%
# Limit of detection

print('Paccar')
lod(Paccar, Augusta, threshold)
print('Reference') 
lod_reference = lod(Reference, Augusta, threshold)
#%%

# just used for testing what impact removing values below the lod has on the calibrations
Reference = Reference[Reference['PM2_5'] > lod_reference]
#%%
Augusta_All.to_csv('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/All_overlap1.csv', index=False)
#%%
Paccar = Paccar.dropna()
Reference = Reference.dropna()
#%%
#Augusta.to_csv('/Users/matthew/Desktop/Augusta.csv', index=False)
#Reference.to_csv('/Users/matthew/Desktop/Reference.csv', index=False)
#Paccar.to_csv('/Users/matthew/Desktop/Paccar.csv', index=False)

#Augusta.to_csv('/Users/matthew/Desktop/Augusta_upsample.csv',  index=True , date_format='%Y-%m-%d %H:%M:%S')
#Reference.to_csv('/Users/matthew/Desktop/Reference_15_min.csv',  index=True , date_format='%Y-%m-%d %H:%M:%S')
#%%
Paccar.to_csv('/Users/matthew/Desktop/Paccar_clarity_adj.csv',  index=True , date_format='%Y-%m-%d %H:%M:%S')
#%%
Reference.to_csv('/Users/matthew/Desktop/Reference_clarity_adj.csv',  index=True , date_format='%Y-%m-%d %H:%M:%S')
#%%
# dates check out for Augusta BAM (continuous dates)

# why is there a  date mismatch between Augusta and Paccar???

# multi linear regression calibration fit (with relative humidity and temperature)
# For Paccar

Paccar['Augusta_PM2_5'] = Augusta['PM2_5']

Paccar = Paccar.dropna()
X = Paccar[['PM2_5','Rel_humid', 'temp']] ##'spec_humid_unitless X usually means our input variables (or independent variables)
###X = Paccar[['Augusta_PM2_5']] ###
#X = X.dropna()
###y = Paccar['PM2_5'] ## Y usually means our output/dependent variable ###
y = Paccar['Augusta_PM2_5']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)
#%%
# Multiple linear Regression
Paccar['mlrPredictions'] = predictions

# Clarity calibrations
#Paccar['mlrPredictions'] = (Paccar['PM2_5']*0.32 + 2.166)   # overall Reference Clarity correction (to check if they got switched on accident)
#Paccar['mlrPredictions'] = (Paccar['PM2_5']*0.48 + 0.95)   # Overall Clarity correction
#Paccar['mlrPredictions'] = (Paccar['PM2_5']*0.334 + 1.616)   # Dec 18 2019 to Jan 31 2020 Clarity correction
#Paccar['mlrPredictions'] = (Paccar['PM2_5']*0.359 + 2.232)   # Feb 1 2020 to Feb 29 2020 Clarity correction

# Minimizing mean error method using just slope: CORRECTION FACTOR IS WRONG FOR THIS, THE CURRENT FACTORS, ie 2.149, 1.861, 2.273 WERE THE MEAN ERRORS NOT SLOPE ADJUSTMENTS
#Paccar['mlrPredictions'] = (Paccar['PM2_5'])/2.149      # Using overall data set (12/18 to 3/5)
#Paccar['mlrPredictions'] = (Paccar['PM2_5'])/1.861      # Using Dec-Jan data set
#Paccar['mlrPredictions'] = (Paccar['PM2_5'])/2.273      # Using Feb data set

# Minimizing mean error method using slope and intercept
#Paccar['mlrPredictions'] = (Paccar['PM2_5'])*0.3939+1.477      # Using overall data set (12/18 to 3/5)
#Paccar['mlrPredictions'] = (Paccar['PM2_5'])*0.377+0.992      # Using Dec-Jan data set
#Paccar['mlrPredictions'] = (Paccar['PM2_5'])*0.44+1.745     # Using Feb data set

# Minimizing mean error method using slope and intercepts subtract intercept first then divide
#Paccar['mlrPredictions'] = (Paccar['PM2_5'] + 3.750)/2.539           # for overall data set
#Paccar['mlrPredictions'] = (Paccar['PM2_5'] + 2.269)/2.651            # for Dec-Jan data set
#Paccar['mlrPredictions'] = (Paccar['PM2_5'] + 3.962)/2.271           # for Feb data set

# our Calibration
#Paccar['mlrPredictions'] = (Paccar['PM2_5'] + 0.8256)/1.9127        # (Paccar['PM2_5'] + 0.5693)/1.9712         ### ACTUAL CORRECTION(Paccar['PM2_5'] + 0.8256)/1.9127 

Paccar['residuals'] = Paccar['Augusta_PM2_5'] - Paccar['PM2_5']
Paccar['prediction_residuals'] = Paccar['Augusta_PM2_5'] - Paccar['mlrPredictions']
#Paccar['predictions_check'] = Paccar['PM2_5']*0.405178+1.2128
Paccar['Location'] = 'Paccar'

#########################################################################

# Von's method for error estimation

# Raw data comparison parameters

n = len(Paccar['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Paccar['S_x'] = Paccar['Augusta_PM2_5']/(sigma_i**2)
S_x = Paccar['S_x'].sum()
Paccar['S_y'] = Paccar['PM2_5']/(sigma_i**2)
S_y = Paccar['S_y'].sum()
Paccar['S_xx'] = (Paccar['Augusta_PM2_5']**2)/(sigma_i**2)
S_xx = Paccar['S_xx'].sum()
Paccar['S_xy'] = ((Paccar['Augusta_PM2_5']*Paccar['PM2_5'])/sigma_i**2)
S_xy = Paccar['S_xy'].sum()
delta = S*S_xx - (S_x)**2
a = ((S_xx*S_y) - (S_x*S_xy))/delta
b = ((S*S_xy) - (S_x*S_y))/delta
var_a = S_xx/delta
var_b = S/delta
stdev_a = var_a**0.5
stdev_b = var_b**0.5
se_a = stdev_a/(n**0.5)
se_b = stdev_b/(n**0.5)
r_ab = (-1*S_x)/((S*S_xx)**0.5)

print('Raw Paccar a =', a, '\n',
      'Raw Paccar b =', b, '\n')

print('Raw Paccar var a =', var_a, '\n',
      'Raw Paccar var b =', var_b, '\n')

print('Raw Paccar standard dev a =', stdev_a, '\n',
      'Raw Paccar standard dev b =', stdev_b, '\n')

print('Raw Paccar standard error a =', se_a, '\n',
      'Raw Paccar standard error b =', se_b, '\n',
      'Raw Paccar r value =', r_ab)


#%%
gaussian_fit(Paccar)
#%%##################################################
# For Reference

Reference['Augusta_PM2_5'] = Augusta['PM2_5']
Reference = Reference.dropna()
X = Reference[['PM2_5','Rel_humid', 'temp']] ##'spec_humid_unitless' X usually means our input variables (or independent variables)
###X = Reference[['Augusta_PM2_5']]
###X = X.dropna()
###y = Reference['PM2_5'] ## Y usually means our output/dependent variable
y = Reference['Augusta_PM2_5']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)
#%%
# plot raw Reference node data to confirm the linear fit
linear_plot(Reference.Augusta_PM2_5, Reference.mlrPredictions, 
            Reference.Augusta_PM2_5, Reference.mlrPredictions,'Raw Reference', 1)   # used for plotting the corrected data with 1 equation for each region



#%%

# Multiple Linear Regression 
Reference['mlrPredictions'] = predictions

# Claritys calibrations
#Reference['mlrPredictions'] = (Reference['PM2_5']*0.48 + 0.95)       # the Paccar overall Clarity Calibration (to check if they got reversed on accident)
#Reference['mlrPredictions'] = (Reference['PM2_5']*0.32 + 2.166)     # Overall Clarity Calibration
#Reference['mlrPredictions'] = (Reference['PM2_5']*0.525 - 0.305)     # Dec 18 2019 to Jan 31 2020 Clarity Calibration
#Reference['mlrPredictions'] = (Reference['PM2_5']*0.491 + 1.569)    # Feb 1 2020 to Feb 29 2020 Clarity Calibration

# Minimizing mean error method using slope and intercepts
#Reference['mlrPredictions'] = (Reference['PM2_5'])*0.433+1.414      # Using overall data set (12/18 to 3/5)
#Reference['mlrPredictions'] = (Reference['PM2_5'])*0.407+0.988      # Using Dec-Jan data set
#Reference['mlrPredictions'] = (Reference['PM2_5'])*0.462+1.774     # Using Feb data set

# Minimizing mean error method using slope and intercepts subtract intercept first then divide
#Reference['mlrPredictions'] = (Reference['PM2_5'] + 3.263)/2.308       # for overall data set       
#Reference['mlrPredictions'] = (Reference['PM2_5'] + 2.429)/2.459       # for Dec-Jan data set   
#Reference['mlrPredictions'] = (Reference['PM2_5'] + 3.845)/2.167       # for Feb data set   

# Our Calibration

#Reference['mlrPredictions'] = (Reference['PM2_5'] + 0.6232)/1.7588       # Actual Calibration

Reference['residuals'] = Reference['Augusta_PM2_5'] - Reference['PM2_5']
Reference['prediction_residuals'] = Reference['Augusta_PM2_5'] - Reference['mlrPredictions']
Reference['Location'] = 'Reference'

#########################################################################

# Von's method for error estimation

# Raw data comparison parameters

n = len(Reference['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Reference['S_x'] = Reference['Augusta_PM2_5']/(sigma_i**2)
S_x = Reference['S_x'].sum()
Reference['S_y'] = Reference['PM2_5']/(sigma_i**2)
S_y = Reference['S_y'].sum()
Reference['S_xx'] = (Reference['Augusta_PM2_5']**2)/(sigma_i**2)
S_xx = Reference['S_xx'].sum()
Reference['S_xy'] = ((Reference['Augusta_PM2_5']*Reference['PM2_5'])/sigma_i**2)
S_xy = Reference['S_xy'].sum()
delta = S*S_xx - (S_x)**2
a = ((S_xx*S_y) - (S_x*S_xy))/delta
b = ((S*S_xy) - (S_x*S_y))/delta
var_a = S_xx/delta
var_b = S/delta
stdev_a = var_a**0.5
stdev_b = var_b**0.5
se_a = stdev_a/(n**0.5)
se_b = stdev_b/(n**0.5)
r_ab = (-1*S_x)/((S*S_xx)**0.5)

print('Ref a =', a, '\n',
      'Ref b =', b, '\n')

print('Ref var a =', var_a, '\n',
      'Ref var b =', var_b, '\n')

print('Ref standard dev a =', stdev_a, '\n',
      'Ref standard dev b =', stdev_b, '\n')

print('Ref standard error a =', se_a, '\n',
      'Ref standard error b =', se_b, '\n',
      'Ref r value =', r_ab)
#%%
gaussian_fit(Reference)
linear_plot(Reference.Augusta_PM2_5, Reference.mlrPredictions, Reference.Augusta_PM2_5, Reference.mlrPredictions, 'Reference', 1)
#%%
print('max BAM value = ', Augusta['PM2_5'].max())
#%%

#Compare Clarity Units to Augusta SRCAA BAM as time series


PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_overlap_mean_resample.html')
    #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_overlap_pad_resample.html')
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(Augusta.index,     Augusta.PM2_5,  legend='Augusta',       color='green',     line_width=2)
#p1.line(Paccar.index,     Paccar.PM2_5,  legend='Paccar',       color='blue',     line_width=2)
#p1.line(Paccar.index,     Paccar.lower_bound, legend='lower limit', color='red', line_width=2)
#p1.line(Paccar.index,     Paccar.upper_bound, legend='upper limit', color='red', line_width=2)
p1.line(Reference_hybrid_calibrated.index,     Reference_hybrid_calibrated.PM2_5_corrected,  legend='Reference',       color='red',     line_width=2)

#For plotting the corrected data compared to Augusta
#p1.line(Augusta.index,     Augusta.PM2_5,  legend='Augusta',       color='gray',  line_alpha = 0.4,   line_width=2) # 
#p1.line(Paccar.index,     Paccar.mlrPredictions,  legend='Paccar Corrected',    line_alpha = 0.8,   color='black',     line_width=2)
#p1.line(Reference.index,     Reference.mlrPredictions,  legend='Reference Corrected',   line_alpha = 0.4,    color='red',     line_width=2)

# For plotting one at a time and for original/adjusted data comparison
#p1.line(Paccar.index,     Paccar.PM2_5,  legend='Paccar Raw Data',    line_alpha = 0.8,   color='gold',     line_width=2)
#p1.line(Reference.index,     Reference.PM2_5,  legend='Reference Raw Data',   line_alpha = 0.8,    color='gold',     line_width=2)
#p1.line(Paccar.index,     Paccar.mlrPredictions,  legend='Paccar Corrected',    line_alpha = 0.8,   color='gold',     line_width=2)
#p1.line(Reference.index,     Reference.mlrPredictions,  legend='Reference Corrected',   line_alpha = 0.8,    color='gold',     line_width=2)
#p1.line(Augusta.index,     Augusta.PM2_5,  legend='Augusta',       color='black',  line_alpha = 0.6,   line_width=2)

tab1 = Panel(child=p1, title="Augusta BAM and Clarity Comparison")

tabs = Tabs(tabs=[ tab1])

show(tabs)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# linear regression calibration fit


import pandas as pd

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy 
from bokeh.layouts import column
import holoviews as hv
import numpy as np
from bokeh.io import export_png
from bokeh.models import HoverTool

df = pd.DataFrame()
df['time'] = Augusta['time']
df['Augusta'] = Augusta['PM2_5']
df['Reference'] = Reference['PM2_5']
df['Paccar'] = Paccar['PM2_5']
df = df.dropna()


PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_pad_resample.html')
    output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_mean_resample.html')
PlotType = 'HTMLfile'

#the data
x=np.array(df.Augusta)
y=np.array(df.Augusta)

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

# For Paccar

#the data
x1=np.array(df.Augusta)
y1=np.array(df.Paccar) 
slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x1, y1)
r_squared1 = r_value11**2

# determine best fit line
par = np.polyfit(x1, y1, 1, full=True)
slope1=par[0][0]
intercept1=par[0][1]
y1_predicted = [slope1*i + intercept1  for i in x1]

# For Reference

#the data
x2=np.array(df.Augusta)
y2=np.array(df.Reference)
slope22, intercept22, r_value22, p_value22, std_err22 = scipy.stats.linregress(x2, y2)
r_squared2 = r_value22**2

# determine best fit line
par = np.polyfit(x2, y2, 1, full=True)
slope2=par[0][0]
intercept2=par[0][1]
y2_predicted = [slope2*i + intercept2  for i in x2]

# plot it
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Clarity Nodes (ug/m^3)')

p1.circle(x,y,legend='BAM 1 to 1 line', color='red')
p1.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

p1.circle(df.Augusta, df.Paccar, legend='Wired 2', color='blue')
p1.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

p1.circle(df.Augusta, df.Reference, legend='Reference', color='green')
p1.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))

p1.legend.location='top_left'
p1.toolbar.logo = None
p1.toolbar_location = None

export_png(p1, filename= '/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_vs_wired_clarity_scatter_mean_resample.png')
#export_png(p1, filename= '/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_vs_wired_clarity_scatter_pad_resample.png')

tab1 = Panel(child=p1, title="SRCAA BAM vs Clarity Comparison")

#Plotting Clarity Relative Humidity vs Augusta BAM PM 2.5

p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Rel Humidity (%)',
            y_axis_label='Clarity Reference PM 2.5 (ug/m3)',
            background_fill_color='#440154',
            match_aspect=True,
            tools="wheel_zoom,reset,pan")


#p2.circle(Augusta.PM2_5, Paccar.Rel_humid, size = 3, color = 'green', legend = 'Paccar')
#p2.circle(Paccar.PM2_5, Paccar.Rel_humid, size = 3, color = 'green', legend = 'Paccar')
#p2.circle(Augusta.PM2_5, Reference.Rel_humid, size = 3, color = 'blue', legend = 'Reference')
#p2.circle(Reference.Rel_humid, Reference.PM2_5, size = 3, color = 'blue', legend = 'Reference')
p2.grid.visible = False
r, bins = p2.hexbin(Reference.Rel_humid, Reference.PM2_5, size=2, hover_color="pink", hover_alpha=0.8)
p2.circle(Reference.Rel_humid, Reference.PM2_5, color="white", size=1)

p2.add_tools(HoverTool(
    tooltips=[("count", "@c"), ("(q,r)", "(@q, @r)")],
    mode="mouse", point_policy="follow_mouse", renderers=[r]
))

p2.legend.location='top_left'


tab2 = Panel(child=p2, title = 'Rel Humid vs Clarity Ref PM 2.5')

# Plotting Clarity temp vs Augusta BAM PM2.5

p3 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Temp (ºC)')

#p3.circle(Augusta.PM2_5, Paccar.temp, size = 3, color = 'green', legend = 'Paccar')
p3.circle(Paccar.PM2_5, Paccar.temp, size = 3, color = 'green', legend = 'Paccar')
#p3.circle(Augusta.PM2_5, Reference.temp, size = 3, color = 'blue', legend = 'Reference')
p3.circle(Reference.PM2_5, Reference.temp, size = 3, color = 'blue', legend = 'Reference')
p3.legend.location='top_left'

tab3 = Panel(child=p3, title = 'Temp vs Augusta BAM')

#plotting multiple linear regression predictions

# Paccar
#the data
x4=np.array(Paccar.Augusta_PM2_5)
y4=np.array(Paccar.mlrPredictions) 
slope44, intercept44, r_value44, p_value44, std_err44 = scipy.stats.linregress(x4, y4)
r_squared4 = r_value44**2
# determine best fit line
par = np.polyfit(x4, y4, 1, full=True)
slope4=par[0][0]
intercept4=par[0][1]
y4_predicted = [slope4*i + intercept4  for i in x4]

# Check with stats model output

X = Paccar[['Augusta_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Paccar['mlrPredictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

#sigma_i = 2.51*2       # Hourly MLR calibrated standard deviation
sigma_i = 1.26*2       # 24 hrs MLR calibrated standard deviation

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Paccar['S_x'] = Paccar['Augusta_PM2_5']/(sigma_i**2)
S_x = Paccar['S_x'].sum()
Paccar['S_y'] = Paccar['mlrPredictions']/(sigma_i**2)
S_y = Paccar['S_y'].sum()
Paccar['S_xx'] = (Paccar['Augusta_PM2_5']**2)/(sigma_i**2)
S_xx = Paccar['S_xx'].sum()
Paccar['S_xy'] = ((Paccar['mlrPredictions']*Paccar['Augusta_PM2_5'])/sigma_i**2)
S_xy = Paccar['S_xy'].sum()
delta = S*S_xx - (S_x)**2
a = ((S_xx*S_y) - (S_x*S_xy))/delta
b = ((S*S_xy) - (S_x*S_y))/delta
var_a = S_xx/delta
var_b = S/delta
stdev_a = var_a**0.5
stdev_b = var_b**0.5
se_a = stdev_a/(n**0.5)
se_b = stdev_b/(n**0.5)
r_ab = (-1*S_x)/((S*S_xx)**0.5)

print('Paccar a =', a, '\n',
      'Paccar b =', b, '\n')

print('Paccar var a =', var_a, '\n',
      'Paccar var b =', var_b, '\n')

print('Paccar standard dev a =', stdev_a, '\n',
      'Paccar standard dev b =', stdev_b, '\n')

print('Paccar standard error a =', se_a, '\n',
      'Paccar standard error b =', se_b, '\n',
      'Paccar r value =', r_ab)

# Reference
#the data
x5=np.array(Reference.Augusta_PM2_5)
y5=np.array(Reference.mlrPredictions) 
slope55, intercept55, r_value55, p_value55, std_err55 = scipy.stats.linregress(x5, y5)
r_squared5 = r_value55**2
# determine best fit line
par = np.polyfit(x5, y5, 1, full=True)
slope5=par[0][0]
intercept5=par[0][1]
y5_predicted = [slope5*i + intercept5  for i in x5]


# Check with stats models

X = Reference[['Augusta_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Reference['mlrPredictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)


#sigma_i = 2.45*2       # hourly MLR calibrated standard deviation
sigma_i = 0.96*2       # 24 hours MLR calibrated standard deviation

# Von's method for error estimation

S = n*(1/(sigma_i**2))
Reference['S_x'] = Reference['Augusta_PM2_5']/(sigma_i**2)
S_x = Reference['S_x'].sum()
Reference['S_y'] = Reference['mlrPredictions']/(sigma_i**2)
S_y = Reference['S_y'].sum()
Reference['S_xx'] = (Reference['Augusta_PM2_5']**2)/(sigma_i**2)
S_xx = Reference['S_xx'].sum()
Reference['S_xy'] = ((Reference['mlrPredictions']*Reference['Augusta_PM2_5'])/sigma_i**2)
S_xy = Reference['S_xy'].sum()
delta = S*S_xx - (S_x)**2
a = ((S_xx*S_y) - (S_x*S_xy))/delta
b = ((S*S_xy) - (S_x*S_y))/delta
var_a = S_xx/delta
var_b = S/delta
stdev_a = var_a**0.5
stdev_b = var_b**0.5
se_a = stdev_a/(n**0.5)
se_b = stdev_b/(n**0.5)
r_ab = (-1*S_x)/((S*S_xx)**0.5)

print('Ref a =', a, '\n',
      'Ref b =', b, '\n')

print('Ref var a =', var_a, '\n',
      'Ref var b =', var_b, '\n')

print('Ref standard dev a =', stdev_a, '\n',
      'Ref standard dev b =', stdev_b, '\n')

print('Ref standard error a =', se_a, '\n',
      'Ref standard error b =', se_b, '\n',
      'Ref r value =', r_ab)


p4 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Wired 2 Calibrated PM 2.5 (ug/m3)',
            title = 'SRCAA BAM vs Wired 2 Calibrated')

#p4.circle(x,y,legend='BAM 1 to 1 line', color='red')
#p4.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p4.circle(Paccar.Augusta_PM2_5, Paccar.mlrPredictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted')
p4.line(x4,y4_predicted,color='black',legend='y='+str(round(slope4,2))+'x+'+str(round(intercept4,2))+ '  ' + 'r^2 = ' + str(round(r_squared4,3)))


#p4.circle(Paccar.mlrPredictions,     Paccar.lower_bound, legend='lower limit', color='red', line_width=2)
#p4.circle(Paccar.mlrPredictions,     Paccar.upper_bound, legend='upper limit', color='red', line_width=2)
#p4.circle(Paccar.mlrPredictions,     Paccar.forecast_lb, legend='lower limit', color='black', line_width=2)
#p4.circle(Paccar.mlrPredictions,     Paccar.forecast_ub, legend='upper limit', color='black', line_width=2)
#p4.circle(x2,Paccar.predictions_check, color='purple', legend='prediction check')

#p4.circle(Augusta.PM2_5, Reference.mlrPredictions, size = 3, color = 'blue', legend = 'Reference')
p4.legend.location='top_left'
#p4.toolbar.logo = None
#p4.toolbar_location = None

tab4 = Panel(child=p4, title = 'Wired 2 Calibrated vs Augusta BAM')


p5 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Wired 2 Predictions (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Wired 2 Calibrated vs Wired 2 Calibrated Residuals')

p5.circle(Paccar.mlrPredictions, Paccar.prediction_residuals, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')

#p5.legend.location='top_left'
#p5.toolbar.logo = None
#p5.toolbar_location = None

tab5 = Panel(child=p5, title = 'Paccar Calibrated Residuals')


p6 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
           y_axis_label='Clarity Reference Calibrated PM 2.5 (ug/m3)',
            title = 'SRCAA BAM vs Reference Calibrated')

#p7.circle(x,y,legend='BAM 1 to 1 line', color='red')
#p6.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p6.circle(Reference.Augusta_PM2_5, Reference.mlrPredictions, size = 3, color = 'gray')#, legend = 'Reference Adjusted')
p6.line(x5,y5_predicted,color='black',legend='y='+str(round(slope5,2))+'x+'+str(round(intercept5,2))+ '  ' + 'r^2 = ' + str(round(r_squared5,3)))

p6.legend.location='top_left'
#p6.toolbar.logo = None
#p6.toolbar_location = None

tab6 = Panel(child=p6, title = 'Reference Calibrated vs Augusta BAM')


p7 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Reference Calibrated (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Reference Calibrated vs Reference Calibrated Residuals')

p7.circle(Reference.mlrPredictions, Reference.prediction_residuals, size = 3, color = 'gray')#, legend = 'Reference Adjusted Residuals')

#p7.legend.location='top_left'
#p7.toolbar.logo = None
#p7.toolbar_location = None

tab7 = Panel(child=p7, title = 'Reference Calibrated Residuals')

p9 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Reference PM 2.5 (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Reference vs Residuals')

p9.circle(Paccar.PM2_5, Paccar.residuals, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')



p10 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Wired 2 PM 2.5 (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Wired 2 vs Residuals')

p10.circle(Reference.PM2_5, Reference.residuals, size = 3, color = 'gray')#, legend = 'Reference Adjusted Residuals')

#p5 = figure(plot_width=900,
#            plot_height=450,
#            x_axis_type='datetime',
#            x_axis_label='Time (local)',
#           y_axis_label='Error (ug/m3)',
#            title = 'Paccar Error')

#p5.line(Paccar.index, Paccar.error, color='blue',legend='Paccar Error')

#p5.line(Paccar.index, Paccar.mlrPredictions, color='black',legend='mlr Predictions', line_width=2)
#p5.line(Paccar.index,     Paccar.lower_bound, legend='lower limit', color='red', line_width=1)
#p5.line(Paccar.index,     Paccar.upper_bound, legend='upper limit', color='red', line_width=1)


#p4.circle(x2,Paccar.predictions_check, color='purple', legend='prediction check')

#p4.circle(Augusta.PM2_5, Reference.mlrPredictions, size = 3, color = 'blue', legend = 'Reference')
#p5.legend.location='top_left'
#p5.toolbar.logo = None
#p5.toolbar_location = None

#tab5 = Panel(child=p5, title = 'Paccar Error')

p8 = gridplot([[p4,p5, p9], [p6, p7, p10]], plot_width = 500, plot_height = 300)
tab8 = Panel(child=p8, title = 'Augusta Residuals and Predictions')

export_png(p8, filename='/Users/matthew/Desktop/data/Augusta_residuals_and_predictions.png')


# Plots 11 and 12 are for stacking the original two regressions (1 from Paccar and 1 from Reference) on top of each other
p11 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Wired 2 (ug/m3)',
            title = 'BAM vs Paccar')

#p11.circle(x,y,legend='BAM 1 to 1 line', color='red')
p11.line(x,y_predicted,color='red',legend='BAM 1 to 1 line')#'y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

p11.circle(df.Augusta, df.Paccar, legend='Paccar', color='gray')
p11.line(x1,y1_predicted,color='black',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

p11.legend.location='top_left'
p11.toolbar.logo = None
p11.toolbar_location = None

p12 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Reference (ug/m3)',
            title = 'BAM vs Reference')

#p12.circle(x,y,legend='BAM 1 to 1 line', color='red')
p12.line(x,y_predicted,color='red',legend='BAM 1 to 1 line')#'y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

p12.circle(df.Augusta, df.Reference, legend='Reference', color='gray')
p12.line(x2,y2_predicted,color='black',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))

p12.legend.location='top_left'
p12.toolbar.logo = None
p12.toolbar_location = None


#tab9 = Panel(child=p13, title = 'BAM vs Raw Clarity Data')

#export_png(p13, filename='/Users/matthew/Desktop/data/BAM_vs_Clarity_side_by_side.png')

tabs = Tabs(tabs=[ tab1,  tab2, tab3, tab8])#, tab9,tab5])

show(tabs)

#%%

# recreate plots that Clarity sent 

p4 = figure(plot_width=300,
            plot_height=500,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Paccar (ug/m3)',
            title = 'SRCAA BAM vs Paccar',
            y_range=(0, 65))

#p4.circle(x,y,legend='BAM 1 to 1 line', color='red')
p4.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

p4.circle(Paccar.Augusta_PM2_5, Paccar.mlrPredictions, size = 3, color = 'blue', legend = 'Paccar Calibrated' + ' ' + 'r^2 = ' + str(round(r_squared4,3)))
p4.legend.label_text_font_size = '6pt'
p4.circle(df.Augusta, df.Paccar, legend='Paccar Raw', color='green')
#p4.line(x4,y4_predicted,color='black',legend='y='+str(round(slope4,2))+'x+'+str(round(intercept4,2))+ '  ' + 'r^2 = ' + str(round(r_squared4,3)))


#p4.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

#p4.circle(df.Augusta, df.Reference, legend='Reference', color='green')
#p4.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))

#p4.circle(Augusta.PM2_5, Reference.mlrPredictions, size = 3, color = 'blue', legend = 'Reference')

p4.legend.location='top_right'
#p4.toolbar.logo = None
#p4.toolbar_location = None

tab4 = Panel(child=p4, title = 'Paccar vs Augusta BAM')

p5 = figure(plot_width=300,
            plot_height=500,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Reference (ug/m3)',
            title = 'SRCAA BAM vs Reference',
            y_range=(0, 60))

p5.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p5.legend.label_text_font_size = '8pt'
p5.circle(Reference.Augusta_PM2_5, Reference.mlrPredictions, size = 3, color = 'blue', legend = 'Reference Calibrated')
p5.circle(df.Augusta, df.Reference, legend='Reference Raw' + '  ' + 'r^2 = ' + str(round(r_squared2,3)), color='green')
#p5.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))



p5.legend.location='top_right'
#p5.toolbar.logo = None
#p5.toolbar_location = None

tab5 = Panel(child=p5, title = 'Reference vs Augusta BAM')

tabs = Tabs(tabs=[ tab4,tab5])#, tab5])

show(tabs)



#%%

# Mean Error Calc and performance stats

print('Augusta average = ', np.mean(Augusta['PM2_5']), '\n')
print('Augusta median = ', np.median(Augusta['PM2_5']), '\n')
print('Augusta sum = ', np.sum(Augusta['PM2_5']), '\n')

print('Paccar Adj average = ', np.mean(Paccar['mlrPredictions']), '\n')
print('Paccar Adj median = ', np.median(Paccar['mlrPredictions']), '\n')
print('Paccar Adj sum = ', np.sum(Paccar['mlrPredictions']), '\n')

print('Reference Adj average = ', np.mean(Reference['mlrPredictions']), '\n')
print('Reference Adj median = ', np.median(Reference['mlrPredictions']), '\n')
print('Reference Adj sum = ', np.sum(Reference['mlrPredictions']), '\n')

print('Paccar Adj RMSE = ', np.sqrt((np.sum((Paccar['mlrPredictions']-Paccar['Augusta_PM2_5'])**2)/len(Paccar['mlrPredictions']))), '\n')
print('Reference Adj RMSE = ', np.sqrt((np.sum((Reference['mlrPredictions']-Reference['Augusta_PM2_5'])**2)/len(Reference['mlrPredictions']))), '\n')


Paccar['total_error'] = abs(Paccar['Augusta_PM2_5']-Paccar['mlrPredictions'])
Paccar_error_sum = Paccar['total_error'].sum()
Paccar_mean_error = Paccar_error_sum/(Paccar['total_error'].count())

print('Paccar error sum = ', Paccar_error_sum, '\n',
      'Paccar mean absolute error =', Paccar_mean_error, '\n')

Paccar_res_over_5 = abs(Paccar['prediction_residuals']).values
Paccar_res_over_5 = Paccar_res_over_5[Paccar_res_over_5 >= 5]

count_over_5 = len(Paccar_res_over_5)

total_count = len(Paccar['mlrPredictions'])

fraction_over = count_over_5/total_count
fraction_under = 1 - fraction_over
print('Paccar Percentage of residuals over 5 ug/m3 = ', fraction_over)
print('Paccar Percentage of residuals under 5ug/m3 = ', fraction_under)
#%%
Reference['total_error'] = abs(Reference['Augusta_PM2_5']-Reference['mlrPredictions'])
Reference_error_sum = Reference['total_error'].sum()
Reference_mean_error = Reference_error_sum/(Reference['total_error'].count())
Reference['total'] = abs(Reference['prediction_residuals'])
total = Reference['total'].sum()
print(total)
total1 = total/1896
print(total1)

print('Ref error sum = ', Reference_error_sum, '\n',
      'Ref mean absolute error =', Reference_mean_error, '\n')

Reference_res_over_5 = abs(Reference['prediction_residuals']).values
Reference_res_over_5 = Reference_res_over_5[Reference_res_over_5 >= 5]

count_over_5 = len(Reference_res_over_5)

total_count = len(Reference['mlrPredictions'])

fraction_over = count_over_5/total_count
fraction_under = 1 - fraction_over
print('Reference Percentage of residuals over 5 ug/m3 = ', fraction_over)
print('Reference Percentage of residuals under 5ug/m3 = ', fraction_under)

#%%
# calc minimized average error by adjusting slope for paccar

def f(x):
    n = Paccar['PM2_5'].count()
    error = abs(Paccar['Augusta_PM2_5'] - (Paccar['PM2_5']*x))#-0.8256))
    error_sum = error.sum()
    err = error_sum/n
    print(x)
   # print(error_sum)
    print(err)
    return(err)
    #return (Paccar['Augusta_PM2_5'] - Paccar['PM2_5']*x).sum())/(Paccar['PM2_5'].count()))

def f2(x):
    n = Reference['PM2_5'].count()
    error = abs(Reference['Augusta_PM2_5'] - (Reference['PM2_5']*x))#-0.6232))
    error_sum = error.sum()
    err = error_sum/n
    print(x)
   # print(error_sum)
    print(err)
    return(err)

#%%
# calc minimized average error using slope and intercept

def f3(params):
    x, y = params
    n = Paccar['PM2_5'].count()
    error = abs(Paccar['Augusta_PM2_5'] - (Paccar['PM2_5']*x+y))
    error_sum = error.sum()
    err = error_sum/n
    print(x)
    print(y)
   # print(error_sum)
    print(err)
    return(err)

def f4(params):
    x, y = params
    n = Reference['PM2_5'].count()
    error = abs(Reference['Augusta_PM2_5'] - (Reference['PM2_5']*x+y))
    error_sum = error.sum()
    err = error_sum/n
    print(x)
    print(y)
   # print(error_sum)
    print(err)
    return(err)
    
#%%
    
# calc minimized average error using slope and intercept and subtracting intercept before multiplying

def f5(params):
    x, y = params
    n = Paccar['PM2_5'].count()
    error = abs(Paccar['Augusta_PM2_5'] - ((Paccar['PM2_5']-y)/x))
    error_sum = error.sum()
    err = error_sum/n
    print(x)
    print(y)
   # print(error_sum)
    print(err)
    return(err)

def f6(params):
    x, y = params
    n = Reference['PM2_5'].count()
    error = abs(Reference['Augusta_PM2_5'] - ((Reference['PM2_5']-y)/x))
    error_sum = error.sum()
    err = error_sum/n
    print(x)
    print(y)
   # print(error_sum)
    print(err)
    return(err)
#%%
from scipy import optimize    
#def g(x):
 #   return x**2 - 1

minimum = optimize.fmin(f, 1)
minimum = optimize.fmin(f2, 1)
#%%
initial_guess = [1,1]

minimum  = optimize.fmin(f3, initial_guess)
minimum  = optimize.fmin(f4, initial_guess)
#%%
initial_guess = [1,1]

minimum  = optimize.fmin(f5, initial_guess)
minimum  = optimize.fmin(f6, initial_guess)
#%%

### Histograms of PM 2.5 measurement distributions
hv.extension('bokeh', logo=False)
import numpy as np
import holoviews as hv

data = Paccar['PM2_5']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p2 = figure(plot_width = 1500,
            plot_height = 700)

p2 = hv.Histogram((edges, frequencies))
p2 = p2.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'Paccar')

data = Reference['PM2_5']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p3 = figure(plot_width = 1500,
            plot_height = 700)

p3 = hv.Histogram((edges, frequencies))
p3 = p3.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'Reference')

data = Augusta['PM2_5']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p4 = figure(plot_width = 1500,
            plot_height = 700)

p4 = hv.Histogram((edges, frequencies))
p4 = p4.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'SRCAA BAM')

p5 = (p2+p3+p4).cols(3)

hv.save(p5.options(toolbar=None), '/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/histogram' 
        + start_time + '_to_' 
        + end_time + '_mean_resample.png', fmt='png', backend='bokeh')    # works

show(hv.render(p5))

#%%

# load in stevens pressure data (closest to Augusta site) for specific humidity calcs

stevens_bme = pd.DataFrame({})
stevens_bme_json = pd.DataFrame({})
stevens_bme, stevens_bme_json = load_indoor('Stevens', stevens_bme,stevens_bme_json, interval, start_time, end_time)

audubon_bme = pd.DataFrame({})
audubon_bme_json = pd.DataFrame({})
audubon_bme, audubon_bme_json = load_indoor('Audubon', audubon_bme,audubon_bme_json, interval, start_time, end_time)

#%%
#spec_humid(audubon_bme, audubon_bme_json, Paccar)
spec_humid(stevens_bme, stevens_bme_json, Paccar)

#spec_humid(audubon_bme, audubon_bme_json, Reference)
spec_humid(stevens_bme, stevens_bme_json, Reference)
#%%
Paccar.to_csv('/Users/matthew/Desktop/spec_humid_test.csv', index=False)
#%%
start_time = '2020-02-16 00:00'
end_time = '2020-03-05 23:00'
Paccar = Paccar.loc[start_time:end_time]

#%%

# Hexbin plot of specific humidity vs PM2.5

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Specific Humidity (kg/kg)',
            y_axis_label='Clarity Paccar PM 2.5 (ug/m3)',
            background_fill_color='#440154',
            match_aspect=True)#,
         #   tools="wheel_zoom,reset,pan")


p1.grid.visible = False
r, bins = p1.hexbin(Paccar.spec_humid_unitless, Paccar.PM2_5, size=1, hover_color="pink", hover_alpha=0.8)
p1.circle(Paccar.spec_humid_unitless, Paccar.PM2_5, color="white", size=1)

p1.add_tools(HoverTool(
    tooltips=[("count", "@c"), ("(q,r)", "(@q, @r)")],
    mode="mouse", point_policy="follow_mouse", renderers=[r]
))

p1.legend.location='top_left'


tab1 = Panel(child=p1, title = 'Specfiic Humid vs Clarity Paccar PM 2.5')

            
export_png(p1, filename='/Users/matthew/Desktop/Augusta_specific_humidity_Clarity_Paccar.png')


p2 = figure(title = 'Specific Humidity',
            plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Specific Humidity (g/kg)',
            y_axis_label='PM 2.5 (ug/m3)')

p2.scatter(Paccar.spec_humid_unitless,       Paccar.mlrPredictions,       legend='Paccar_Stevens_Pressure',            color='black',     line_width=2)        


p2.legend.location='top_left'

tab2 = Panel(child=p2, title="Specific Humidity")

p3 = figure(title = 'Specific Humidity vs Relative Humidity',
            plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Specific Humidity (g/kg)',
            y_axis_label='Relative Humidity (%)')

p3.scatter(Paccar.spec_humid_unitless,       Paccar.Rel_humid,                 color='black',     line_width=2)        


p3.legend.location='top_left'

tab23= Panel(child=p3, title="Specific Humidity vs Rel Humidity")

tabs = Tabs(tabs=[ tab1, tab2])

show(tabs)

#%%

p3 = figure(title = 'Specific Humidity vs Relative Humidity',
            plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Specific Humidity (g/kg)',
            y_axis_label='Relative Humidity (%)')

p3.scatter(Paccar.spec_humid_unitless,       Paccar.Rel_humid,                 color='black',     line_width=2)        


p3.legend.location='top_left'

tab3= Panel(child=p3, title="Specific Humidity vs Rel Humidity")

tabs = Tabs(tabs=[ tab3])

show(tabs)


#%%

# Calc limit of detection

