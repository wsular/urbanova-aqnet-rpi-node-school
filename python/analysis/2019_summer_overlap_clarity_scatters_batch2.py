#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:13:54 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:35:16 2020

@author: matthew
"""
import pandas as pd
from glob import glob
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from bokeh.io import export_png
import numpy as np
import scipy 
from bokeh.layouts import row
from bokeh.layouts import gridplot
from linear_plot_function import linear_plot
from gaussian_fit_function import gaussian_fit
#%%

Adams_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Adams*.csv')
files.sort()
for file in files:
    Adams_All = pd.concat([Adams_All, pd.read_csv(file)], sort=False)
    
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
    
Paccar_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Paccar*.csv')
files.sort()
for file in files:
    Paccar_All = pd.concat([Paccar_All, pd.read_csv(file)], sort=False)



#%%

start_time = '2019-08-19 00:00'
end_time = '2019-08-22 23:00'

batch_1_start = '2019-12-17 15:00'
batch_1_end = '2019-12-17 15:00'

batch_2_start = '2019-12-17 15:00'
batch_2_end = '2019-12-17 15:00'

interval = '15T'
#%%

### Batch 2 scatters and time series

import statsmodels.api as sm
import statistics


Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]
Reference = Reference.resample(interval).mean()
    
Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar = Paccar_All.loc[start_time:end_time]
Paccar = Paccar.resample(interval).mean()
Paccar['Ref_PM2_5'] = Reference['PM2_5']
Paccar = Paccar.dropna()

#X = Paccar[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Paccar[['Ref_PM2_5']]
#X = X.dropna()
y = Paccar['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

Paccar['Predictions'] = (Paccar['PM2_5']-0.1975)/1.0286
Paccar['residuals'] = Paccar['Ref_PM2_5'] - Paccar['PM2_5']
Paccar['prediction_residuals'] = Paccar['Ref_PM2_5'] - Paccar['Predictions']

n = len(Paccar['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Paccar['S_x'] = Paccar['Ref_PM2_5']/(sigma_i**2)
S_x = Paccar['S_x'].sum()
Paccar['S_y'] = Paccar['PM2_5']/(sigma_i**2)
S_y = Paccar['S_y'].sum()
Paccar['S_xx'] = (Paccar['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Paccar['S_xx'].sum()
Paccar['S_xy'] = ((Paccar['Ref_PM2_5']*Paccar['PM2_5'])/sigma_i**2)
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
      'Paccar r vale =', r_ab)


Adams_All['time'] = pd.to_datetime(Adams_All['time'])
Adams_All = Adams_All.sort_values('time')
Adams_All.index = Adams_All.time
Adams = Adams_All.loc[start_time:end_time]    
Adams = Adams.resample(interval).pad()
Adams['Ref_PM2_5'] = Reference['PM2_5']
Adams = Adams.dropna()
# take out Clarity node measurements below LOD to see impact on linear calibration results (ie slope, int and uncertainties in those)
Adams = Adams[Adams['PM2_5'] > 4.87]

#X = Adams[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Adams[['Ref_PM2_5']]
#X = X.dropna()
y = Adams['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

# linear correction if using all calibration data
#Adams['Predictions'] = (Adams['PM2_5']+0.93)/1.1554
# linear correction if using LOD limited data
Adams['Predictions'] = (Adams['PM2_5']-1.4203)/1.08

Adams['residuals'] = Adams['Ref_PM2_5'] - Adams['PM2_5']
Adams['prediction_residuals'] = Adams['Ref_PM2_5'] - Adams['Predictions']

#Adams.to_csv('/Users/matthew/Desktop/adamsTest.csv', index=False)
n = len(Adams['PM2_5'])
sigma_i = 1.82        # 1 standard deviation from residuals from linear adjustment to reference Clarity node
S = n*(1/(sigma_i**2))
Adams['S_x'] = Adams['Ref_PM2_5']/(sigma_i**2)
S_x = Adams['S_x'].sum()
Adams['S_y'] = Adams['PM2_5']/(sigma_i**2)
S_y = Adams['S_y'].sum()
Adams['S_xx'] = (Adams['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Adams['S_xx'].sum()
Adams['S_xy'] = ((Adams['Ref_PM2_5']*Adams['PM2_5'])/sigma_i**2)
S_xy = Adams['S_xy'].sum()
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

print('Adams a =', a, '\n',
      'Adams b =', b, '\n')

print('Adams var a =', var_a, '\n',
      'Adams var b =', var_b, '\n')

print('Adams standard dev a =', stdev_a, '\n',
      'Adams standard dev b =', stdev_b, '\n')

print('Adams standard error a =', se_a, '\n',
      'Adams standard error b =', se_b, '\n',
      'Adams r vale =', r_ab)


Grant_All['time'] = pd.to_datetime(Grant_All['time'])
Grant_All = Grant_All.sort_values('time')
Grant_All.index = Grant_All.time
Grant = Grant_All.loc[start_time:end_time]
Grant = Grant.resample(interval).pad()
Grant['Ref_PM2_5'] = Reference['PM2_5']
Grant = Grant.dropna()
# take out Clarity node measurements below LOD to see impact on linear calibration results (ie slope, int and uncertainties in those)
Grant = Grant[Grant['PM2_5'] > 4.87]

#X = Grant[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Grant[['Ref_PM2_5']]
#X = X.dropna()
y = Grant['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

# linear correction if using all calibration data
#Grant['Predictions'] = (Grant['PM2_5']+1.0965)/1.29
# linear correction if using LOD limited data
Grant['Predictions'] = (Grant['PM2_5']-1.5206)/1.21

Grant['residuals'] = Grant['Ref_PM2_5'] - Grant['PM2_5']
Grant['prediction_residuals'] = Grant['Ref_PM2_5'] - Grant['Predictions']

n = len(Grant['PM2_5'])
sigma_i = 1.52
S = n*(1/(sigma_i**2))
Grant['S_x'] = Grant['Ref_PM2_5']/(sigma_i**2)
S_x = Grant['S_x'].sum()
Grant['S_y'] = Grant['PM2_5']/(sigma_i**2)
S_y = Grant['S_y'].sum()
Grant['S_xx'] = (Grant['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Grant['S_xx'].sum()
Grant['S_xy'] = ((Grant['Ref_PM2_5']*Grant['PM2_5'])/sigma_i**2)
S_xy = Grant['S_xy'].sum()
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

print('Grant a =', a, '\n',
      'Grant b =', b, '\n')

print('Grant var a =', var_a, '\n',
      'Grant var b =', var_b, '\n')

print('Grant standard dev a =', stdev_a, '\n',
      'Grant standard dev b =', stdev_b, '\n')

print('Grant standard error a =', se_a, '\n',
      'Grant standard error b =', se_b, '\n',
      'Grant r vale =', r_ab)

Jefferson_All['time'] = pd.to_datetime(Jefferson_All['time'])
Jefferson_All = Jefferson_All.sort_values('time')
Jefferson_All.index = Jefferson_All.time
Jefferson = Jefferson_All.loc[start_time:end_time]
Jefferson = Jefferson.resample(interval).pad()
Jefferson['Ref_PM2_5'] = Reference['PM2_5']
Jefferson = Jefferson.dropna()
# take out Clarity node measurements below LOD to see impact on linear calibration results (ie slope, int and uncertainties in those)
Jefferson = Jefferson[Jefferson['PM2_5'] > 4.87]


#X = Jefferson[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Jefferson[['Ref_PM2_5']]
#X = X.dropna()
y = Jefferson['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

# linear correction if using all calibration data
#Jefferson['Predictions'] = (Jefferson['PM2_5']+0.7099)/1.1458
# linear correction if using LOD limited data
Jefferson['Predictions'] = (Jefferson['PM2_5']-2.75)/1.04

Jefferson['residuals'] = Jefferson['Ref_PM2_5'] - Jefferson['PM2_5']
Jefferson['prediction_residuals'] = Jefferson['Ref_PM2_5'] - Jefferson['Predictions']

n = len(Jefferson['PM2_5'])
sigma_i = 1.79
S = n*(1/(sigma_i**2))
Jefferson['S_x'] = Jefferson['Ref_PM2_5']/(sigma_i**2)
S_x = Jefferson['S_x'].sum()
Jefferson['S_y'] = Jefferson['PM2_5']/(sigma_i**2)
S_y = Jefferson['S_y'].sum()
Jefferson['S_xx'] = (Jefferson['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Jefferson['S_xx'].sum()
Jefferson['S_xy'] = ((Jefferson['Ref_PM2_5']*Jefferson['PM2_5'])/sigma_i**2)
S_xy = Jefferson['S_xy'].sum()
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

print('Jefferson a =', a, '\n',
      'Jefferson b =', b, '\n')

print('Jefferson var a =', var_a, '\n',
      'Jefferson var b =', var_b, '\n')

print('Jefferson standard dev a =', stdev_a, '\n',
      'Jefferson standard dev b =', stdev_b, '\n')

print('Jefferson standard error a =', se_a, '\n',
      'Jefferson standard error b =', se_b, '\n',
      'Jefferson r vale =', r_ab)


Sheridan_All['time'] = pd.to_datetime(Sheridan_All['time'])
Sheridan_All = Sheridan_All.sort_values('time')
Sheridan_All.index = Sheridan_All.time
Sheridan = Sheridan_All.loc[start_time:end_time]
Sheridan = Sheridan.resample(interval).pad()
Sheridan['Ref_PM2_5'] = Reference['PM2_5']
Sheridan = Sheridan.dropna()
# take out Clarity node measurements below LOD to see impact on linear calibration results (ie slope, int and uncertainties in those)
Sheridan = Sheridan[Sheridan['PM2_5'] > 4.87]


#X = Sheridan[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Sheridan[['Ref_PM2_5']]
#X = X.dropna()
y = Sheridan['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

# linear correction if using all calibration data
#Sheridan['Predictions'] = (Sheridan['PM2_5']+0.6958)/1.1468
# linear correction if using LOD limited data
Sheridan['Predictions'] = (Sheridan['PM2_5']-2.92)/1.03

Sheridan['residuals'] = Sheridan['Ref_PM2_5'] - Sheridan['PM2_5']
Sheridan['prediction_residuals'] = Sheridan['Ref_PM2_5'] - Sheridan['Predictions']

n = len(Sheridan['PM2_5'])
sigma_i = 1.78
S = n*(1/(sigma_i**2))
Sheridan['S_x'] = Sheridan['Ref_PM2_5']/(sigma_i**2)
S_x = Sheridan['S_x'].sum()
Sheridan['S_y'] = Sheridan['PM2_5']/(sigma_i**2)
S_y = Sheridan['S_y'].sum()
Sheridan['S_xx'] = (Sheridan['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Sheridan['S_xx'].sum()
Sheridan['S_xy'] = ((Sheridan['Ref_PM2_5']*Sheridan['PM2_5'])/sigma_i**2)
S_xy = Sheridan['S_xy'].sum()
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

print('Sheridan a =', a, '\n',
      'Sheridan b =', b, '\n')

print('Sheridan var a =', var_a, '\n',
      'Sheridan var b =', var_b, '\n')

print('Sheridan standard dev a =', stdev_a, '\n',
      'Sheridan standard dev b =', stdev_b, '\n')

print('Sheridan standard error a =', se_a, '\n',
      'Sheridan standard error b =', se_b, '\n',
      'Sheridan r vale =', r_ab)

Stevens_All['time'] = pd.to_datetime(Stevens_All['time'])
Stevens_All = Stevens_All.sort_values('time')
Stevens_All.index = Stevens_All.time
Stevens = Stevens_All.loc[start_time:end_time]
Stevens = Stevens.resample(interval).pad()
Stevens['Ref_PM2_5'] = Reference['PM2_5']
Stevens = Stevens.dropna()
# take out Clarity node measurements below LOD to see impact on linear calibration results (ie slope, int and uncertainties in those)
Stevens = Stevens[Stevens['PM2_5'] > 4.87]

#X = Stevens[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Stevens[['Ref_PM2_5']]
#X = X.dropna()
y = Stevens['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

# linear correction if using all calibration data
#Stevens['Predictions'] = (Stevens['PM2_5']+0.8901)/1.2767
# linear correction if using LOD limited data
Stevens['Predictions'] = (Stevens['PM2_5']-2.1848)/1.18

Stevens['residuals'] = Stevens['Ref_PM2_5'] - Stevens['PM2_5']
Stevens['prediction_residuals'] = Stevens['Ref_PM2_5'] - Stevens['Predictions']

n = len(Stevens['PM2_5'])
sigma_i = 1.73
S = n*(1/(sigma_i**2))
Stevens['S_x'] = Stevens['Ref_PM2_5']/(sigma_i**2)
S_x = Stevens['S_x'].sum()
Stevens['S_y'] = Stevens['PM2_5']/(sigma_i**2)
S_y = Stevens['S_y'].sum()
Stevens['S_xx'] = (Stevens['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Stevens['S_xx'].sum()
Stevens['S_xy'] = ((Stevens['Ref_PM2_5']*Stevens['PM2_5'])/sigma_i**2)
S_xy = Stevens['S_xy'].sum()
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

print('Stevens a =', a, '\n',
      'Stevens b =', b, '\n')

print('Stevens var a =', var_a, '\n',
      'Stevens var b =', var_b, '\n')

print('Stevens standard dev a =', stdev_a, '\n',
      'Stevens standard dev b =', stdev_b, '\n')

print('Stevens standard error a =', se_a, '\n',
      'Stevens standard error b =', se_b, '\n',
      'Stevens r vale =', r_ab)


#%%

# Fit Gaussian to residual plots

gaussian_fit(Adams)
gaussian_fit(Grant)
gaussian_fit(Jefferson)
gaussian_fit(Sheridan)
gaussian_fit(Stevens)


#%%
PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/Clarity_wired_overlap_batch2_pad_resample.html')
    #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_overlap_pad_resample.html')
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        

p1.line(Paccar.index,     Paccar.PM2_5,  legend='Paccar',       color='blue',     line_width=2)
p1.line(Reference.index,     Reference.PM2_5,  legend='Reference',       color='red',     line_width=2)
p1.line(Adams.index,     Adams.PM2_5,  legend='Adams',       color='yellow',     line_width=2)
p1.line(Grant.index,     Grant.PM2_5,  legend='Grant',       color='green',     line_width=2)
p1.line(Jefferson.index,     Jefferson.PM2_5,  legend='Jefferson',       color='purple',     line_width=2)
p1.line(Sheridan.index,     Sheridan.PM2_5,  legend='Sheridan',       color='orange',     line_width=2)
p1.line(Stevens.index,     Stevens.PM2_5,  legend='Stevens',       color='brown',     line_width=2)

p1.legend.location='top_left'
p1.toolbar.logo = None
p1.toolbar_location = None


tab1 = Panel(child=p1, title="Clarity Comparison")



# tab for plotting scatter plots of clarity nodes vs clarity "Reference" wired node
df = pd.DataFrame()
df['Reference'] = Reference['PM2_5']
df['Adams'] = Adams['PM2_5']
df['Paccar'] = Paccar['PM2_5']
df['Grant'] = Grant['PM2_5']
df['Jefferson'] = Jefferson['PM2_5']
df['Sheridan'] = Sheridan['PM2_5']
df['Stevens'] = Stevens['PM2_5']
df = df.dropna()

#the data for Reference 1 to 1 line
x=np.array(df.Reference)
y=np.array(df.Reference)

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

#the data for Reference vs Adams
x1=np.array(df.Reference)
y1=np.array(df.Adams) 
slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x1, y1)
r_squared1 = r_value11**2
# determine best fit line
par = np.polyfit(x1, y1, 1, full=True)
slope1=par[0][0]
intercept1=par[0][1]
y1_predicted = [slope1*i + intercept1  for i in x1]

#the data for Reference vs Grant
x2=np.array(df.Reference)
y2=np.array(df.Grant)
slope22, intercept22, r_value22, p_value22, std_err22 = scipy.stats.linregress(x2, y2)
r_squared2 = r_value22**2

# determine best fit line
par = np.polyfit(x2, y2, 1, full=True)
slope2=par[0][0]
intercept2=par[0][1]
y2_predicted = [slope2*i + intercept2  for i in x2]


#the data for Reference vs Jefferson
x3=np.array(df.Reference)
y3=np.array(df.Jefferson)
slope33, intercept33, r_value33, p_value33, std_err33 = scipy.stats.linregress(x3, y3)
r_squared3 = r_value33**2

# determine best fit line
par = np.polyfit(x3, y3, 1, full=True)
slope3=par[0][0]
intercept3=par[0][1]
y3_predicted = [slope3*i + intercept3  for i in x3]


#the data for Reference vs Sheridan
x4=np.array(df.Reference)
y4=np.array(df.Sheridan)
slope44, intercept44, r_value44, p_value44, std_err44 = scipy.stats.linregress(x4, y4)
r_squared4 = r_value44**2

# determine best fit line
par = np.polyfit(x4, y4, 1, full=True)
slope4=par[0][0]
intercept4=par[0][1]
y4_predicted = [slope4*i + intercept4  for i in x4]

#the data for Reference vs Stevens
x5=np.array(df.Reference)
y5=np.array(df.Stevens)
slope55, intercept55, r_value55, p_value55, std_err55 = scipy.stats.linregress(x5, y5)
r_squared5 = r_value55**2

# determine best fit line
par = np.polyfit(x5, y5, 1, full=True)
slope5=par[0][0]
intercept5=par[0][1]
y5_predicted = [slope5*i + intercept5  for i in x5]


#the data for Reference vs Paccar
x6=np.array(df.Reference)
y6=np.array(df.Paccar)
slope66, intercept66, r_value66, p_value66, std_err66 = scipy.stats.linregress(x6, y6)
r_squared6 = r_value66**2

# determine best fit line
par = np.polyfit(x6, y6, 1, full=True)
slope6=par[0][0]
intercept6=par[0][1]
y6_predicted = [slope6*i + intercept6  for i in x6]


#### For prediction Plotting ##########

#the data for Reference vs Adams
x11=np.array(Adams.Ref_PM2_5)
y11=np.array(Adams.Predictions) 
slope111, intercept111, r_value111, p_value111, std_err111 = scipy.stats.linregress(x11, y11)
r_squared11 = r_value111**2
# determine best fit line
par = np.polyfit(x11, y11, 1, full=True)
slope11=par[0][0]
intercept11=par[0][1]
y11_predicted = [slope11*i + intercept11  for i in x11]

# Check with stats model output

X = Adams[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Adams['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 1.82*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Adams['S_x'] = Adams['Ref_PM2_5']/(sigma_i**2)
S_x = Adams['S_x'].sum()
Adams['S_y'] = Adams['Predictions']/(sigma_i**2)
S_y = Adams['S_y'].sum()
Adams['S_xx'] = (Adams['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Adams['S_xx'].sum()
Adams['S_xy'] = ((Adams['Predictions']*Adams['Ref_PM2_5'])/sigma_i**2)
S_xy = Adams['S_xy'].sum()
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

print('Adams a =', a, '\n',
      'Adams b =', b, '\n')

print('Adams var a =', var_a, '\n',
      'Adams var b =', var_b, '\n')

print('Adams standard dev a =', stdev_a, '\n',
      'Adams standard dev b =', stdev_b, '\n')

print('Adams standard error a =', se_a, '\n',
      'Adams standard error b =', se_b, '\n',
      'Adams r value =', r_ab)


#the data for Reference vs Grant
x22=np.array(Grant.Ref_PM2_5)
y22=np.array(Grant.Predictions)
slope222, intercept222, r_value222, p_value222, std_err222 = scipy.stats.linregress(x22, y22)
r_squared22 = r_value222**2

# determine best fit line
par = np.polyfit(x22, y22, 1, full=True)
slope22=par[0][0]
intercept22=par[0][1]
y22_predicted = [slope22*i + intercept22  for i in x22]

# Check with stats model output

X = Grant[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Grant['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 1.52*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Grant['S_x'] = Grant['Ref_PM2_5']/(sigma_i**2)
S_x = Grant['S_x'].sum()
Grant['S_y'] = Grant['Predictions']/(sigma_i**2)
S_y = Grant['S_y'].sum()
Grant['S_xx'] = (Grant['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Grant['S_xx'].sum()
Grant['S_xy'] = ((Grant['Predictions']*Grant['Ref_PM2_5'])/sigma_i**2)
S_xy = Grant['S_xy'].sum()
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

print('Grant a =', a, '\n',
      'Grant b =', b, '\n')

print('Grant var a =', var_a, '\n',
      'Grant var b =', var_b, '\n')

print('Grant standard dev a =', stdev_a, '\n',
      'Grant standard dev b =', stdev_b, '\n')

print('Grant standard error a =', se_a, '\n',
      'Grant standard error b =', se_b, '\n',
      'Grant r value =', r_ab)


#the data for Reference vs Jefferson
x33=np.array(Jefferson.Ref_PM2_5)
y33=np.array(Jefferson.Predictions)
slope333, intercept333, r_value333, p_value333, std_err333 = scipy.stats.linregress(x33, y33)
r_squared33 = r_value333**2

# determine best fit line
par = np.polyfit(x33, y33, 1, full=True)
slope33=par[0][0]
intercept33=par[0][1]
y33_predicted = [slope33*i + intercept33  for i in x33]

# Check with stats model output

X = Jefferson[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Jefferson['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 1.79*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Jefferson['S_x'] = Jefferson['Ref_PM2_5']/(sigma_i**2)
S_x = Jefferson['S_x'].sum()
Jefferson['S_y'] = Jefferson['Predictions']/(sigma_i**2)
S_y = Jefferson['S_y'].sum()
Jefferson['S_xx'] = (Jefferson['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Jefferson['S_xx'].sum()
Jefferson['S_xy'] = ((Jefferson['Predictions']*Jefferson['Ref_PM2_5'])/sigma_i**2)
S_xy = Jefferson['S_xy'].sum()
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

print('Jefferson a =', a, '\n',
      'Jefferson b =', b, '\n')

print('Jefferson var a =', var_a, '\n',
      'Jefferson var b =', var_b, '\n')

print('Jefferson standard dev a =', stdev_a, '\n',
      'Jefferson standard dev b =', stdev_b, '\n')

print('Jefferson standard error a =', se_a, '\n',
      'Jefferson standard error b =', se_b, '\n',
      'Jefferson r value =', r_ab)

#the data for Reference vs Sheridan
x44=np.array(Sheridan.Ref_PM2_5)
y44=np.array(Sheridan.Predictions)
slope444, intercept444, r_value444, p_value444, std_err444 = scipy.stats.linregress(x44, y44)
r_squared44 = r_value444**2

# determine best fit line
par = np.polyfit(x44, y44, 1, full=True)
slope44=par[0][0]
intercept44=par[0][1]
y44_predicted = [slope44*i + intercept44  for i in x44]

# Check with stats model output

X = Sheridan[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Sheridan['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 1.78*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Sheridan['S_x'] = Sheridan['Ref_PM2_5']/(sigma_i**2)
S_x = Sheridan['S_x'].sum()
Sheridan['S_y'] = Sheridan['Predictions']/(sigma_i**2)
S_y = Sheridan['S_y'].sum()
Sheridan['S_xx'] = (Sheridan['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Sheridan['S_xx'].sum()
Sheridan['S_xy'] = ((Sheridan['Predictions']*Sheridan['Ref_PM2_5'])/sigma_i**2)
S_xy = Sheridan['S_xy'].sum()
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

print('Sheridan a =', a, '\n',
      'Sheridan b =', b, '\n')

print('Sheridan var a =', var_a, '\n',
      'Sheridan var b =', var_b, '\n')

print('Sheridan standard dev a =', stdev_a, '\n',
      'Sheridan standard dev b =', stdev_b, '\n')

print('Sheridan standard error a =', se_a, '\n',
      'Sheridan standard error b =', se_b, '\n',
      'Sheridan r value =', r_ab)

#the data for Reference vs Stevens
x55=np.array(Stevens.Ref_PM2_5)
y55=np.array(Stevens.Predictions)
slope555, intercept555, r_value555, p_value555, std_err555 = scipy.stats.linregress(x55, y55)
r_squared55 = r_value555**2

# determine best fit line
par = np.polyfit(x55, y55, 1, full=True)
slope55=par[0][0]
intercept55=par[0][1]
y55_predicted = [slope55*i + intercept55  for i in x55]

# Check with stats model output

X = Stevens[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Stevens['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 1.73*2
# Von's method for error estimation
S = n*(1/(sigma_i**2))
Stevens['S_x'] = Stevens['Ref_PM2_5']/(sigma_i**2)
S_x = Stevens['S_x'].sum()
Stevens['S_y'] = Stevens['Predictions']/(sigma_i**2)
S_y = Stevens['S_y'].sum()
Stevens['S_xx'] = (Stevens['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Stevens['S_xx'].sum()
Stevens['S_xy'] = ((Stevens['Predictions']*Stevens['Ref_PM2_5'])/sigma_i**2)
S_xy = Stevens['S_xy'].sum()
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

print('Stevens a =', a, '\n',
      'Stevens b =', b, '\n')

print('Stevens var a =', var_a, '\n',
      'Stevens var b =', var_b, '\n')

print('Stevens standard dev a =', stdev_a, '\n',
      'Stevens standard dev b =', stdev_b, '\n')

print('Stevens standard error a =', se_a, '\n',
      'Stevens standard error b =', se_b, '\n',
      'Stevens r value =', r_ab)


#the data for Reference vs Paccar
x66=np.array(Paccar.Ref_PM2_5)
y66=np.array(Paccar.Predictions)
slope666, intercept666, r_value666, p_value666, std_err666 = scipy.stats.linregress(x66, y66)
r_squared66 = r_value666**2

# determine best fit line
par = np.polyfit(x66, y66, 1, full=True)
slope66=par[0][0]
intercept66=par[0][1]
y66_predicted = [slope66*i + intercept66  for i in x66]

# Check with stats model output

X = Paccar[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Paccar['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Paccar['S_x'] = Paccar['Ref_PM2_5']/(sigma_i**2)
S_x = Paccar['S_x'].sum()
Paccar['S_y'] = Paccar['Predictions']/(sigma_i**2)
S_y = Paccar['S_y'].sum()
Paccar['S_xx'] = (Paccar['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Paccar['S_xx'].sum()
Paccar['S_xy'] = ((Paccar['Predictions']*Paccar['Ref_PM2_5'])/sigma_i**2)
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

# plot it
p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Clarity Nodes(ug/m^3)')

p2.circle(x,y,legend='Reference 1 to 1 line', color='red')
p2.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

#p2.circle(df.Reference, df.Audubon, legend='Audubon', color='blue')
#p2.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

#p2.circle(df.Reference, df.Balboa, legend='Balboa', color='green')
#p2.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))

#p2.circle(df.Reference, df.Browne, legend='Browne', color='yellow')
#p2.line(x3,y3_predicted,color='yellow',legend='y='+str(round(slope3,2))+'x+'+str(round(intercept3,2))+ '  ' + 'r^2 = ' + str(round(r_squared3,3)))

#p2.circle(df.Reference, df.Lidgerwood, legend='Lidgerwood', color='brown')
#p2.line(x4,y4_predicted,color='brown',legend='y='+str(round(slope4,2))+'x+'+str(round(intercept4,2))+ '  ' + 'r^2 = ' + str(round(r_squared4,3)))

#p2.circle(df.Reference, df.Regal, legend='Regal', color='purple')
#p2.line(x5,y5_predicted,color='purple',legend='y='+str(round(slope5,2))+'x+'+str(round(intercept5,2))+ '  ' + 'r^2 = ' + str(round(r_squared5,3)))

#p2.circle(df.Reference, df.Paccar, legend='Paccar', color='magenta')
#p2.line(x6,y6_predicted,color='magenta',legend='y='+str(round(slope6,2))+'x+'+str(round(intercept6,2))+ '  ' + 'r^2 = ' + str(round(r_squared6,3)))

p2.legend.location='top_left'
p2.toolbar.logo = None
p2.toolbar_location = None


p3 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Adams (ug/m^3)')

p3.circle(df.Reference, df.Adams, legend='Adams', color='black')
p3.legend.label_text_font_size = "10px"
p3.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

p3.legend.location='top_left'
p3.toolbar.logo = None
p2.toolbar_location = None

p4 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Grant (ug/m^3)')

p4.circle(df.Reference, df.Grant, legend='Grant', color='black')
p4.line(x2,y2_predicted,color='black',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))
p4.legend.label_text_font_size = "10px"

p4.legend.location='top_left'
p4.toolbar.logo = None
p2.toolbar_location = None

p5 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Jefferson (ug/m^3)')

p5.circle(df.Reference, df.Jefferson, legend='Jefferson', color='black')
p5.line(x3,y3_predicted,color='black',legend='y='+str(round(slope3,2))+'x+'+str(round(intercept3,2))+ '  ' + 'r^2 = ' + str(round(r_squared3,3)))
p5.legend.label_text_font_size = "10px"

p5.legend.location='top_left'
p5.toolbar.logo = None
p5.toolbar_location = None

p6 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Sheridan (ug/m^3)')

p6.circle(df.Reference, df.Sheridan, legend='Sheridan', color='black')
p6.line(x4,y4_predicted,color='black',legend='y='+str(round(slope4,2))+'x+'+str(round(intercept4,2))+ '  ' + 'r^2 = ' + str(round(r_squared4,3)))
p6.legend.label_text_font_size = "10px"

p6.legend.location='top_left'
p6.toolbar.logo = None
p6.toolbar_location = None

p7 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Stevens (ug/m^3)')

p7.circle(df.Reference, df.Stevens, legend='Stevens', color='black')
p7.line(x5,y5_predicted,color='black',legend='y='+str(round(slope5,2))+'x+'+str(round(intercept5,2))+ '  ' + 'r^2 = ' + str(round(r_squared5,3)))
p7.legend.label_text_font_size = "10px"

p7.legend.location='top_left'
p7.toolbar.logo = None
p7.toolbar_location = None

p8 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Paccar (ug/m^3)')

p8.circle(df.Reference, df.Paccar, legend='Paccar', color='black')
p8.line(x6,y6_predicted,color='black',legend='y='+str(round(slope6,2))+'x+'+str(round(intercept6,2))+ '  ' + 'r^2 = ' + str(round(r_squared6,3)))
p8.legend.label_text_font_size = "10px"

p8.legend.location='top_left'
p8.toolbar.logo = None
p8.toolbar_location = None

p9 = gridplot([[p3,p4, p5], [p6, p7, p8]], plot_width = 400, plot_height = 300)


tab2 = Panel(child=p9, title="Clarity Scatter Comparison")

p10 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Adams Adjusted (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Adams Adjusted vs Adams Residuals')

#p10.circle(Audubon.PM2_5, Audubon.residuals, size = 3, color = 'gray')#, legend = 'Audubon Residuals')
p10.circle(Adams.Predictions, Adams.prediction_residuals, size = 3, color = 'gray')#, legend = 'Audubon Residuals')

p11 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Grant Adjusted (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Grant Adjusted vs Grant Residuals')

#p11.circle(Balboa.PM2_5, Balboa.residuals, size = 3, color = 'gray')#, legend = 'Balboa Residuals')
p11.circle(Grant.Predictions, Grant.prediction_residuals, size = 3, color = 'gray')#, legend = 'Balboa Residuals')

p12 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Jefferson Adjusted (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Jefferson Adjusted vs Jefferson Residuals')

#p12.circle(Browne.PM2_5, Browne.residuals, size = 3, color = 'gray')#, legend = 'Browne Residuals')
p12.circle(Jefferson.Predictions, Jefferson.prediction_residuals, size = 3, color = 'gray')#, legend = 'Browne Residuals')

p13 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Sheridan Adjusted (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Sheridan Adjusted vs Sheridan Residuals')

#p13.circle(Lidgerwood.PM2_5, Lidgerwood.residuals, size = 3, color = 'gray')#, legend = 'Lidgerwood Residuals')
p13.circle(Sheridan.Predictions, Sheridan.prediction_residuals, size = 3, color = 'gray')#, legend = 'Lidgerwood Residuals')

p14 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Stevens Adjusted (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Stevens Adjusted vs Stevens Residuals')

#p14.circle(Regal.PM2_5, Regal.residuals, size = 3, color = 'gray')#, legend = 'Regal Residuals')
p14.circle(Stevens.Predictions, Stevens.prediction_residuals, size = 3, color = 'gray')#, legend = 'Regal Residuals')

p15 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Paccar Adjusted (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Paccar vs Paccar Residuals')

#p15.circle(Paccar.PM2_5, Paccar.residuals, size = 3, color = 'gray')#, legend = 'Paccar Residuals')
p15.circle(Paccar.Predictions, Paccar.prediction_residuals, size = 3, color = 'gray')#, legend = 'Paccar Residuals')

p16 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Adams Predictions (ug/m3)',
            title = 'Clarity Reference vs Adams Predictions')

p16.circle(Adams.Ref_PM2_5, Adams.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
p16.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p16.line(x11,y11_predicted,color='black',legend='y='+str(round(slope11,2))+'x+'+str(round(intercept11,2))+ '  ' + 'r^2 = ' + str(round(r_squared11,3)))

p16.legend.location='top_left'


p17 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Grant Predictions (ug/m3)',
            title = 'Clarity Reference vs Grant Predictions')

p17.circle(Grant.Ref_PM2_5, Grant.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
p17.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p17.line(x22,y22_predicted,color='black',legend='y='+str(round(slope22,2))+'x+'+str(round(intercept22,2))+ '  ' + 'r^2 = ' + str(round(r_squared22,3)))

p17.legend.location='top_left'

p18 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Jefferson Predictions (ug/m3)',
            title = 'Clarity Reference vs Jefferson Predictions')

p18.circle(Jefferson.Ref_PM2_5, Jefferson.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
p18.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p18.line(x33,y33_predicted,color='black',legend='y='+str(round(slope33,2))+'x+'+str(round(intercept33,2))+ '  ' + 'r^2 = ' + str(round(r_squared33,3)))

p18.legend.location='top_left'


p19 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Sheridan Predictions (ug/m3)',
            title = 'Clarity Reference vs Sheridan Predictions')

p19.circle(Sheridan.Ref_PM2_5, Sheridan.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
p19.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p19.line(x44,y44_predicted,color='black',legend='y='+str(round(slope44,2))+'x+'+str(round(intercept44,2))+ '  ' + 'r^2 = ' + str(round(r_squared44,3)))

p19.legend.location='top_left'


p20 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Stevens Predictions (ug/m3)',
            title = 'Clarity Reference vs Stevens Predictions')

p20.circle(Stevens.Ref_PM2_5, Stevens.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
p20.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p20.line(x55,y55_predicted,color='black',legend='y='+str(round(slope55,2))+'x+'+str(round(intercept55,2))+ '  ' + 'r^2 = ' + str(round(r_squared55,3)))

p20.legend.location='top_left'


p21 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Paccar Predictions (ug/m3)',
            title = 'Clarity Reference vs Paccar Predictions')

p21.circle(Paccar.Ref_PM2_5, Paccar.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
p21.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p21.line(x66,y66_predicted,color='black',legend='y='+str(round(slope66,2))+'x+'+str(round(intercept66,2))+ '  ' + 'r^2 = ' + str(round(r_squared66,3)))

p21.legend.location='top_left'


p16 = gridplot([[p16,p17, p18], [p10, p11, p12]], plot_width = 400, plot_height = 300)
p22 = gridplot([[p19,p20, p21], [p13, p14, p15]], plot_width = 400, plot_height = 300)


tab3 = Panel(child=p16, title="Clarity Node vs Clarity Reference Comparisons")
tab4 = Panel(child=p22, title="Clarity Node vs Clarity Reference Comparisons")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4])


show(tabs)


export_png(p1, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_2_wired_time_series_pad_resample.png")
export_png(p9, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_2_scatter_pad_resample.png")
    
export_png(p16, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_2_residuals_and_predictions_1.png")    
export_png(p22, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_2_residuals_and_predictions_2.png")  
    


#%%
#def linear_plot(x,y,x_winter,y_winter,unit_name,n_lines,**kwargs):
linear_plot(Grant.Ref_PM2_5, Grant.PM2_5, Grant.Ref_PM2_5, Grant.Predictions, 'Grant', 1)
#%%
linear_plot(Jefferson.Ref_PM2_5, Jefferson.PM2_5, Jefferson.Ref_PM2_5, Jefferson.Predictions, 'Jefferson', 1)
#%%
linear_plot(Adams.Ref_PM2_5, Adams.PM2_5, Adams.Ref_PM2_5, Adams.Predictions, 'Adams', 1)
#%%
linear_plot(Sheridan.Ref_PM2_5, Sheridan.PM2_5, Sheridan.Ref_PM2_5, Sheridan.Predictions, 'Sheridan', 1)
#%%
linear_plot(Stevens.Ref_PM2_5, Stevens.PM2_5, Stevens.Ref_PM2_5, Stevens.Predictions, 'Stevens', 1)
    
    
    
    
    
    
    
    
    