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
from gaussian_fit_function import gaussian_fit
from linear_plot_function import linear_plot
#%%


Audubon_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Audubon*.csv')
files.sort()
for file in files:
    Audubon_All = pd.concat([Audubon_All, pd.read_csv(file)], sort=False)
   
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

start_time = '2019-07-01 00:00'
end_time = '2019-07-31 23:00'

batch_1_start = '2019-12-17 15:00'
batch_1_end = '2019-12-17 15:00'

batch_2_start = '2019-12-17 15:00'
batch_2_end = '2019-12-17 15:00'

interval = '15T'
#%%

### Batch 1 scatters and time series


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
#predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

#Paccar['Predictions'] = predictions
Paccar['Predictions'] = (Paccar['PM2_5']-0.08)/1.1159
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


Audubon_All['time'] = pd.to_datetime(Audubon_All['time'])
Audubon_All = Audubon_All.sort_values('time')
Audubon_All.index = Audubon_All.time
Audubon = Audubon_All.loc[start_time:end_time]   
Audubon = Audubon.resample(interval).mean()
Audubon['Ref_PM2_5'] = Reference['PM2_5']
Audubon['Location'] = 'Audubon'
Audubon = Audubon.dropna()

#X = Audubon[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Audubon[['Ref_PM2_5']]
#X = X.dropna()
y = Audubon['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

Audubon['Predictions'] = (Audubon['PM2_5']-0.36)/1.09
Audubon['residuals'] = Audubon['Ref_PM2_5'] - Audubon['PM2_5']
Audubon['prediction_residuals'] = Audubon['Ref_PM2_5'] - Audubon['Predictions']

n = len(Audubon['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Audubon['S_x'] = Audubon['Ref_PM2_5']/(sigma_i**2)
S_x = Audubon['S_x'].sum()
Audubon['S_y'] = Audubon['PM2_5']/(sigma_i**2)
S_y = Audubon['S_y'].sum()
Audubon['S_xx'] = (Audubon['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Audubon['S_xx'].sum()
Audubon['S_xy'] = ((Audubon['Ref_PM2_5']*Audubon['PM2_5'])/sigma_i**2)
S_xy = Audubon['S_xy'].sum()
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

print('Audubon a =', a, '\n',
      'Audubon b =', b, '\n')

print('Audubon var a =', var_a, '\n',
      'Audubon var b =', var_b, '\n')

print('Audubon standard dev a =', stdev_a, '\n',
      'Audubon standard dev b =', stdev_b, '\n')

print('Audubon standard error a =', se_a, '\n',
      'Audubon standard error b =', se_b, '\n',
      'Audubon r vale =', r_ab)
#%%

Balboa_All['time'] = pd.to_datetime(Balboa_All['time'])
Balboa_All = Balboa_All.sort_values('time')
Balboa_All.index = Balboa_All.time
Balboa = Balboa_All.loc[start_time:end_time]    
Balboa = Balboa.resample(interval).mean()
Balboa['Ref_PM2_5'] = Reference['PM2_5']
Balboa['Location'] = 'Balboa'
Balboa = Balboa.dropna()

Balboa['Ref_PM2_5'] = Reference['PM2_5']
#X = Balboa[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Balboa[['Ref_PM2_5']]
#X = X.dropna()
y = Balboa['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

Balboa['Predictions'] =( Balboa['PM2_5']-0.2878)/1.2457
Balboa['residuals'] = Balboa['Ref_PM2_5'] - Balboa['PM2_5']
Balboa['prediction_residuals'] = Balboa['Ref_PM2_5'] - Balboa['Predictions']

n = len(Balboa['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Balboa['S_x'] = Balboa['Ref_PM2_5']/(sigma_i**2)
S_x = Balboa['S_x'].sum()
Balboa['S_y'] = Balboa['PM2_5']/(sigma_i**2)
S_y = Balboa['S_y'].sum()
Balboa['S_xx'] = (Balboa['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Balboa['S_xx'].sum()
Balboa['S_xy'] = ((Balboa['Ref_PM2_5']*Balboa['PM2_5'])/sigma_i**2)
S_xy = Balboa['S_xy'].sum()
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

print('Balboa a =', a, '\n',
      'Balboa b =', b, '\n')

print('Balboa var a =', var_a, '\n',
      'Balboa var b =', var_b, '\n')

print('Balboa standard dev a =', stdev_a, '\n',
      'Balboa standard dev b =', stdev_b, '\n')

print('Balboa standard error a =', se_a, '\n',
      'Balboa standard error b =', se_b, '\n',
      'Balboa r vale =', r_ab)


Browne_All['time'] = pd.to_datetime(Browne_All['time'])
Browne_All = Browne_All.sort_values('time')
Browne_All.index = Browne_All.time
Browne = Browne_All.loc[start_time:end_time]
Browne = Browne.resample(interval).mean()
Browne['Ref_PM2_5'] = Reference['PM2_5']
Browne['Location'] = 'Browne'
Browne = Browne.dropna()

Browne['Ref_PM2_5'] = Reference['PM2_5']
#X = Browne[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Browne[['Ref_PM2_5']]
#X = X.dropna()
y = Browne['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

Browne['Predictions'] = (Browne['PM2_5']-0.4771)/1.1082
Browne['residuals'] = Browne['Ref_PM2_5'] - Browne['PM2_5']
Browne['prediction_residuals'] = Browne['Ref_PM2_5'] - Browne['Predictions']

n = len(Browne['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Browne['S_x'] = Browne['Ref_PM2_5']/(sigma_i**2)
S_x = Browne['S_x'].sum()
Browne['S_y'] = Browne['PM2_5']/(sigma_i**2)
S_y = Browne['S_y'].sum()
Browne['S_xx'] = (Browne['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Browne['S_xx'].sum()
Browne['S_xy'] = ((Browne['Ref_PM2_5']*Browne['PM2_5'])/sigma_i**2)
S_xy = Browne['S_xy'].sum()
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

print('Browne a =', a, '\n',
      'Browne b =', b, '\n')

print('Browne var a =', var_a, '\n',
      'Browne var b =', var_b, '\n')

print('Browne standard dev a =', stdev_a, '\n',
      'Browne standard dev b =', stdev_b, '\n')

print('Browne standard error a =', se_a, '\n',
      'Browne standard error b =', se_b, '\n',
      'Browne r vale =', r_ab)


Lidgerwood_All['time'] = pd.to_datetime(Lidgerwood_All['time'])
Lidgerwood_All = Lidgerwood_All.sort_values('time')
Lidgerwood_All.index = Lidgerwood_All.time
Lidgerwood = Lidgerwood_All.loc[start_time:end_time]
Lidgerwood = Lidgerwood.resample(interval).mean()
Lidgerwood['Ref_PM2_5'] = Reference['PM2_5']
Lidgerwood['Location'] = 'Lidgerwood'
Lidgerwood = Lidgerwood.dropna()

Lidgerwood['Ref_PM2_5'] = Reference['PM2_5']
#X = Lidgerwood[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Lidgerwood[['Ref_PM2_5']]
#X = X.dropna()
y = Lidgerwood['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

Lidgerwood['Predictions'] = (Lidgerwood['PM2_5']-1.1306)/0.9566
Lidgerwood['residuals'] = Lidgerwood['Ref_PM2_5'] - Lidgerwood['PM2_5']
Lidgerwood['prediction_residuals'] = Lidgerwood['Ref_PM2_5'] - Lidgerwood['Predictions']

n = len(Lidgerwood['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Lidgerwood['S_x'] = Lidgerwood['Ref_PM2_5']/(sigma_i**2)
S_x = Lidgerwood['S_x'].sum()
Lidgerwood['S_y'] = Lidgerwood['PM2_5']/(sigma_i**2)
S_y = Lidgerwood['S_y'].sum()
Lidgerwood['S_xx'] = (Lidgerwood['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Lidgerwood['S_xx'].sum()
Lidgerwood['S_xy'] = ((Lidgerwood['Ref_PM2_5']*Lidgerwood['PM2_5'])/sigma_i**2)
S_xy = Lidgerwood['S_xy'].sum()
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

print('Lidgerwood a =', a, '\n',
      'Lidgerwood b =', b, '\n')

print('Lidgerwood var a =', var_a, '\n',
      'Lidgerwood var b =', var_b, '\n')

print('Lidgerwood standard dev a =', stdev_a, '\n',
      'Lidgerwood standard dev b =', stdev_b, '\n')

print('Lidgerwood standard error a =', se_a, '\n',
      'Lidgerwood standard error b =', se_b, '\n',
      'Lidgerwood r vale =', r_ab)


Regal_All['time'] = pd.to_datetime(Regal_All['time'])
Regal_All = Regal_All.sort_values('time')
Regal_All.index = Regal_All.time
Regal = Regal_All.loc[start_time:end_time]
Regal = Regal.resample(interval).mean()
Regal['Ref_PM2_5'] = Reference['PM2_5']
Regal['Location'] = 'Regal'
Regal = Regal.dropna()

Regal['Ref_PM2_5'] = Reference['PM2_5']
#X = Regal[['PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
X = Regal[['Ref_PM2_5']]
#X = X.dropna()
y = Regal['PM2_5'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

Regal['Predictions'] = (Regal['PM2_5']-0.247)/0.9915
Regal['residuals'] = Regal['Ref_PM2_5'] - Regal['PM2_5']
Regal['prediction_residuals'] = Regal['Ref_PM2_5'] - Regal['Predictions']

n = len(Regal['PM2_5'])
sigma_i = 5
S = n*(1/(sigma_i**2))
Regal['S_x'] = Regal['Ref_PM2_5']/(sigma_i**2)
S_x = Regal['S_x'].sum()
Regal['S_y'] = Regal['PM2_5']/(sigma_i**2)
S_y = Regal['S_y'].sum()
Regal['S_xx'] = (Regal['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Regal['S_xx'].sum()
Regal['S_xy'] = ((Regal['Ref_PM2_5']*Regal['PM2_5'])/sigma_i**2)
S_xy = Regal['S_xy'].sum()
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

print('Regal a =', a, '\n',
      'Regal b =', b, '\n')

print('Regal var a =', var_a, '\n',
      'Regal var b =', var_b, '\n')

print('Regal standard dev a =', stdev_a, '\n',
      'Regal standard dev b =', stdev_b, '\n')

print('Regal standard error a =', se_a, '\n',
      'Regal standard error b =', se_b, '\n',
      'Regal r vale =', r_ab)

#%%

# Fit Gaussian to residual plots

gaussian_fit(Audubon)
gaussian_fit(Balboa)
gaussian_fit(Browne)
gaussian_fit(Lidgerwood)
gaussian_fit(Regal)
#%%
PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/Clarity_wired_overlap_mean_resample.html')
    #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_overlap_pad_resample.html')
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        

p1.line(Paccar.index,     Paccar.PM2_5,  legend='Wired 2',       color='blue',     line_width=2)             # Paccar
p1.line(Reference.index,     Reference.PM2_5,  legend='Reference',       color='red',     line_width=2)     # Reference
p1.line(Audubon.index,     Audubon.PM2_5,  legend='Audubon',       color='yellow',     line_width=2)        # 1
p1.line(Balboa.index,     Balboa.PM2_5,  legend='Balboa',       color='green',     line_width=2)            # 2
p1.line(Browne.index,     Browne.PM2_5,  legend='Browne',       color='purple',     line_width=2)           # 3
p1.line(Lidgerwood.index,     Lidgerwood.PM2_5,  legend='Lidgerwood',       color='orange',     line_width=2) # 4
p1.line(Regal.index,     Regal.PM2_5,  legend='Regal',       color='brown',     line_width=2)            # 5

p1.legend.location = 'top_left'
tab1 = Panel(child=p1, title="Clarity Comparison")


# tab for plotting scatter plots of clarity nodes vs clarity "Reference" wired node

df = pd.DataFrame()
df['Reference'] = Reference['PM2_5']
df['Audubon'] = Audubon['PM2_5']
df['Paccar'] = Paccar['PM2_5']
df['Balboa'] = Balboa['PM2_5']
df['Browne'] = Browne['PM2_5']
df['Lidgerwood'] = Lidgerwood['PM2_5']
df['Regal'] = Regal['PM2_5']
df = df.dropna()

#the data for Reference 1 to 1 line
x=np.array(df.Reference)
y=np.array(df.Reference)

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

#the data for Reference vs Audubon
x1=np.array(df.Reference)
y1=np.array(df.Audubon) 
slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x1, y1)
r_squared1 = r_value11**2
# determine best fit line
par = np.polyfit(x1, y1, 1, full=True)
slope1=par[0][0]
intercept1=par[0][1]
y1_predicted = [slope1*i + intercept1  for i in x1]

#the data for Reference vs Balboa
x2=np.array(df.Reference)
y2=np.array(df.Balboa)
slope22, intercept22, r_value22, p_value22, std_err22 = scipy.stats.linregress(x2, y2)
r_squared2 = r_value22**2

# determine best fit line
par = np.polyfit(x2, y2, 1, full=True)
slope2=par[0][0]
intercept2=par[0][1]
y2_predicted = [slope2*i + intercept2  for i in x2]


#the data for Reference vs Browne
x3=np.array(df.Reference)
y3=np.array(df.Browne)
slope33, intercept33, r_value33, p_value33, std_err33 = scipy.stats.linregress(x3, y3)
r_squared3 = r_value33**2

# determine best fit line
par = np.polyfit(x3, y3, 1, full=True)
slope3=par[0][0]
intercept3=par[0][1]
y3_predicted = [slope3*i + intercept3  for i in x3]


#the data for Reference vs Lidgerwood
x4=np.array(df.Reference)
y4=np.array(df.Lidgerwood)
slope44, intercept44, r_value44, p_value44, std_err44 = scipy.stats.linregress(x4, y4)
r_squared4 = r_value44**2

# determine best fit line
par = np.polyfit(x4, y4, 1, full=True)
slope4=par[0][0]
intercept4=par[0][1]
y4_predicted = [slope4*i + intercept4  for i in x4]

#the data for Reference vs Regal
x5=np.array(df.Reference)
y5=np.array(df.Regal)
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


#########################################

# For Prediction plotting

#the data for Reference vs Balboa
x11=np.array(Audubon.Ref_PM2_5)
y11=np.array(Audubon.Predictions)
slope111, intercept111, r_value111, p_value111, std_err111 = scipy.stats.linregress(x11, y11)
r_squared11 = r_value111**2

# determine best fit line
par = np.polyfit(x11, y11, 1, full=True)
slope11=par[0][0]
intercept11=par[0][1]
y11_predicted = [slope11*i + intercept11  for i in x11]

# Check with stats model output

X = Audubon[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Audubon['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 0.78*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Audubon['S_x'] = Audubon['Ref_PM2_5']/(sigma_i**2)
S_x = Audubon['S_x'].sum()
Audubon['S_y'] = Audubon['Predictions']/(sigma_i**2)
S_y = Audubon['S_y'].sum()
Audubon['S_xx'] = (Audubon['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Audubon['S_xx'].sum()
Audubon['S_xy'] = ((Audubon['Predictions']*Audubon['Ref_PM2_5'])/sigma_i**2)
S_xy = Audubon['S_xy'].sum()
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

print('Audubon a =', a, '\n',
      'Audubon b =', b, '\n')

print('Audubon var a =', var_a, '\n',
      'Audubon var b =', var_b, '\n')

print('Audubon standard dev a =', stdev_a, '\n',
      'Audubon standard dev b =', stdev_b, '\n')

print('Audubon standard error a =', se_a, '\n',
      'Audubon standard error b =', se_b, '\n',
      'Audubon r value =', r_ab)



#the data for Reference vs Balboa
x22=np.array(Balboa.Ref_PM2_5)
y22=np.array(Balboa.Predictions)
slope222, intercept222, r_value222, p_value222, std_err222 = scipy.stats.linregress(x22, y22)
r_squared22 = r_value222**2

# determine best fit line
par = np.polyfit(x22, y22, 1, full=True)
slope22=par[0][0]
intercept22=par[0][1]
y22_predicted = [slope22*i + intercept22  for i in x22]

# Check with stats model output

X = Balboa[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Balboa['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 0.80*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Balboa['S_x'] = Balboa['Ref_PM2_5']/(sigma_i**2)
S_x = Balboa['S_x'].sum()
Balboa['S_y'] = Balboa['Predictions']/(sigma_i**2)
S_y = Balboa['S_y'].sum()
Balboa['S_xx'] = (Balboa['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Balboa['S_xx'].sum()
Balboa['S_xy'] = ((Balboa['Predictions']*Balboa['Ref_PM2_5'])/sigma_i**2)
S_xy = Balboa['S_xy'].sum()
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

print('Balboa a =', a, '\n',
      'Balboa b =', b, '\n')

print('Balboa var a =', var_a, '\n',
      'Balboa var b =', var_b, '\n')

print('Balboa standard dev a =', stdev_a, '\n',
      'Balboa standard dev b =', stdev_b, '\n')

print('Balboa standard error a =', se_a, '\n',
      'Balboa standard error b =', se_b, '\n',
      'Balboa r value =', r_ab)


#the data for Reference vs Browne
x33=np.array(Browne.Ref_PM2_5)
y33=np.array(Browne.Predictions)
slope333, intercept333, r_value333, p_value333, std_err333 = scipy.stats.linregress(x33, y33)
r_squared33 = r_value333**2

# determine best fit line
par = np.polyfit(x33, y33, 1, full=True)
slope33=par[0][0]
intercept33=par[0][1]
y33_predicted = [slope33*i + intercept33  for i in x33]

# Check with stats model output

X = Browne[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Browne['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 0.78*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Browne['S_x'] = Browne['Ref_PM2_5']/(sigma_i**2)
S_x = Browne['S_x'].sum()
Browne['S_y'] = Browne['Predictions']/(sigma_i**2)
S_y = Browne['S_y'].sum()
Browne['S_xx'] = (Browne['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Browne['S_xx'].sum()
Browne['S_xy'] = ((Browne['Predictions']*Browne['Ref_PM2_5'])/sigma_i**2)
S_xy = Browne['S_xy'].sum()
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

print('Browne a =', a, '\n',
      'Browne b =', b, '\n')

print('Browne var a =', var_a, '\n',
      'Browne var b =', var_b, '\n')

print('Browne standard dev a =', stdev_a, '\n',
      'Browne standard dev b =', stdev_b, '\n')

print('Browne standard error a =', se_a, '\n',
      'Browne standard error b =', se_b, '\n',
      'Browne r value =', r_ab)

#the data for Reference vs Lidgerwood
x44=np.array(Lidgerwood.Ref_PM2_5)
y44=np.array(Lidgerwood.Predictions)
slope444, intercept444, r_value444, p_value444, std_err444 = scipy.stats.linregress(x44, y44)
r_squared44 = r_value444**2

# determine best fit line
par = np.polyfit(x44, y44, 1, full=True)
slope44=par[0][0]
intercept44=par[0][1]
y44_predicted = [slope44*i + intercept44  for i in x44]

# Check with stats model output

X = Lidgerwood[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Lidgerwood['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 1.20*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Lidgerwood['S_x'] = Lidgerwood['Ref_PM2_5']/(sigma_i**2)
S_x = Lidgerwood['S_x'].sum()
Lidgerwood['S_y'] = Lidgerwood['Predictions']/(sigma_i**2)
S_y = Lidgerwood['S_y'].sum()
Lidgerwood['S_xx'] = (Lidgerwood['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Lidgerwood['S_xx'].sum()
Lidgerwood['S_xy'] = ((Lidgerwood['Predictions']*Lidgerwood['Ref_PM2_5'])/sigma_i**2)
S_xy = Lidgerwood['S_xy'].sum()
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

print('Lidgerwood a =', a, '\n',
      'Lidgerwood b =', b, '\n')

print('Lidgerwood var a =', var_a, '\n',
      'Lidgerwood var b =', var_b, '\n')

print('Lidgerwood standard dev a =', stdev_a, '\n',
      'Lidgerwood standard dev b =', stdev_b, '\n')

print('Lidgerwood standard error a =', se_a, '\n',
      'Lidgerwood standard error b =', se_b, '\n',
      'Lidgerwood r value =', r_ab)

#the data for Reference vs Regal
x55=np.array(Regal.Ref_PM2_5)
y55=np.array(Regal.Predictions)
slope555, intercept555, r_value555, p_value555, std_err555 = scipy.stats.linregress(x55, y55)
r_squared55 = r_value555**2

# determine best fit line
par = np.polyfit(x55, y55, 1, full=True)
slope55=par[0][0]
intercept55=par[0][1]
y55_predicted = [slope55*i + intercept55  for i in x55]

# Check with stats model output

X = Regal[['Ref_PM2_5']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = Regal['Predictions'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

sigma_i = 0.87*2

# Von's method for error estimation
S = n*(1/(sigma_i**2))
Regal['S_x'] = Regal['Ref_PM2_5']/(sigma_i**2)
S_x = Regal['S_x'].sum()
Regal['S_y'] = Regal['Predictions']/(sigma_i**2)
S_y = Regal['S_y'].sum()
Regal['S_xx'] = (Regal['Ref_PM2_5']**2)/(sigma_i**2)
S_xx = Regal['S_xx'].sum()
Regal['S_xy'] = ((Regal['Predictions']*Regal['Ref_PM2_5'])/sigma_i**2)
S_xy = Regal['S_xy'].sum()
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

print('Regal a =', a, '\n',
      'Regal b =', b, '\n')

print('Regal var a =', var_a, '\n',
      'Regal var b =', var_b, '\n')

print('Regal standard dev a =', stdev_a, '\n',
      'Regal standard dev b =', stdev_b, '\n')

print('Regal standard error a =', se_a, '\n',
      'Regal standard error b =', se_b, '\n',
      'Regal r value =', r_ab)


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
#p2.toolbar_location = None


p3 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Audubon (ug/m^3)')

p3.circle(df.Reference, df.Audubon, legend='Audubon', color='blue')
p3.legend.label_text_font_size = "10px"
p3.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

p3.legend.location='top_left'
p3.toolbar.logo = None
#p2.toolbar_location = None

p4 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Balboa (ug/m^3)')

p4.circle(df.Reference, df.Balboa, legend='Balboa', color='green')
p4.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))
p4.legend.label_text_font_size = "10px"

p4.legend.location='top_left'
p4.toolbar.logo = None
#p2.toolbar_location = None

p5 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Browne (ug/m^3)')

p5.circle(df.Reference, df.Browne, legend='Browne', color='gold')
p5.line(x3,y3_predicted,color='gold',legend='y='+str(round(slope3,2))+'x+'+str(round(intercept3,2))+ '  ' + 'r^2 = ' + str(round(r_squared3,3)))
p5.legend.label_text_font_size = "10px"

p5.legend.location='top_left'
p5.toolbar.logo = None
#p5.toolbar_location = None

p6 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Lidgerwood (ug/m^3)')

p6.circle(df.Reference, df.Lidgerwood, legend='Lidgerwood', color='brown')
p6.line(x4,y4_predicted,color='brown',legend='y='+str(round(slope4,2))+'x+'+str(round(intercept4,2))+ '  ' + 'r^2 = ' + str(round(r_squared4,3)))
p6.legend.label_text_font_size = "10px"

p6.legend.location='top_left'
p6.toolbar.logo = None
#p6.toolbar_location = None

p7 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Regal (ug/m^3)')

p7.circle(df.Reference, df.Regal, legend='Regal', color='purple')
p7.line(x5,y5_predicted,color='purple',legend='y='+str(round(slope5,2))+'x+'+str(round(intercept5,2))+ '  ' + 'r^2 = ' + str(round(r_squared5,3)))
p7.legend.label_text_font_size = "10px"

p7.legend.location='top_left'
p7.toolbar.logo = None
#p7.toolbar_location = None

p8 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Paccar (ug/m^3)')

p8.circle(df.Reference, df.Paccar, legend='Paccar', color='teal')
p8.line(x6,y6_predicted,color='teal',legend='y='+str(round(slope6,2))+'x+'+str(round(intercept6,2))+ '  ' + 'r^2 = ' + str(round(r_squared6,3)))
p8.legend.label_text_font_size = "10px"

p8.legend.location='top_left'
p8.toolbar.logo = None
#p7.toolbar_location = None

p9 = gridplot([[p3,p4, p5], [p6, p7, p8]], plot_width = 400, plot_height = 300)


tab2 = Panel(child=p9, title="Clarity Scatter Comparison")


p10 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Audubon Calibrated (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Audubon vs Audubon Residuals')

#p10.circle(Audubon.PM2_5, Audubon.residuals, size = 3, color = 'gray')#, legend = 'Audubon Residuals')
p10.circle(Audubon.Predictions, Audubon.prediction_residuals, size = 3, color = 'gray')#, legend = 'Audubon Residuals')

p11 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Balboa Calibrated (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Balboa vs Balboa Residuals')

#p11.circle(Balboa.PM2_5, Balboa.residuals, size = 3, color = 'gray')#, legend = 'Balboa Residuals')
p11.circle(Balboa.Predictions, Balboa.prediction_residuals, size = 3, color = 'gray')#, legend = 'Balboa Residuals')

p12 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Browne Calibrated (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Browne vs Browne Residuals')

#p12.circle(Browne.PM2_5, Browne.residuals, size = 3, color = 'gray')#, legend = 'Browne Residuals')
p12.circle(Browne.Predictions, Browne.prediction_residuals, size = 3, color = 'gray')#, legend = 'Browne Residuals')

p13 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Lidgerwood Calibrated (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Lidgerwood vs Lidgerwood Residuals')

#p13.circle(Lidgerwood.PM2_5, Lidgerwood.residuals, size = 3, color = 'gray')#, legend = 'Lidgerwood Residuals')
p13.circle(Lidgerwood.Predictions, Lidgerwood.prediction_residuals, size = 3, color = 'gray')#, legend = 'Lidgerwood Residuals')

p14 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Regal Calibrated (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Regal vs Regal Residuals')

#p14.circle(Regal.PM2_5, Regal.residuals, size = 3, color = 'gray')#, legend = 'Regal Residuals')
p14.circle(Regal.Predictions, Regal.prediction_residuals, size = 3, color = 'gray')#, legend = 'Regal Residuals')

p15 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Paccar Calibrated (ug/m^3)',
            y_axis_label='Residuals (ug/m3)',
            title = 'Paccar vs Paccar Residuals')

#p15.circle(Paccar.PM2_5, Paccar.residuals, size = 3, color = 'gray')#, legend = 'Paccar Residuals')
p15.circle(Paccar.Predictions, Paccar.prediction_residuals, size = 3, color = 'gray')#, legend = 'Paccar Residuals')

p16 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='#1 Calibrated PM 2.5 (ug/m3)',
            title = 'Clarity Reference vs #1')

p16.circle(Audubon.Ref_PM2_5, Audubon.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Calibrated Residuals')
#p16.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p16.line(x11,y11_predicted,color='black',legend='y='+str(round(slope11,2))+'x+'+str(round(intercept11,2))+ '  ' + 'r^2 = ' + str(round(r_squared11,3)))

p16.legend.location='top_left'


p17 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='#2 Calibrated PM 2.5 (ug/m3)',
            title = 'Clarity Reference vs Balboa Predictions')

p17.circle(Balboa.Ref_PM2_5, Balboa.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Calibrated Residuals')
#p17.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p17.line(x22,y22_predicted,color='black',legend='y='+str(round(slope22,2))+'x+'+str(round(intercept22,2))+ '  ' + 'r^2 = ' + str(round(r_squared22,3)))

p17.legend.location='top_left'

p18 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='#3 Calibrated PM 2.5 (ug/m3)',
            title = 'Clarity Reference vs Browne Predictions')

p18.circle(Browne.Ref_PM2_5, Browne.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Calibrated Residuals')
#p18.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p18.line(x33,y33_predicted,color='black',legend='y='+str(round(slope33,2))+'x+'+str(round(intercept33,2))+ '  ' + 'r^2 = ' + str(round(r_squared33,3)))

p18.legend.location='top_left'


p19 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='#4 Calibrated PM 2.5 (ug/m3)',
            title = 'Clarity Reference vs Lidgerwood Predictions')

p19.circle(Lidgerwood.Ref_PM2_5, Lidgerwood.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Calibrated Residuals')
#p19.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p19.line(x44,y44_predicted,color='black',legend='y='+str(round(slope44,2))+'x+'+str(round(intercept44,2))+ '  ' + 'r^2 = ' + str(round(r_squared44,3)))

p19.legend.location='top_left'


p20 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='#5 Calibrated PM 2.5 (ug/m3)',
            title = 'Clarity Reference vs Regal')

p20.circle(Regal.Ref_PM2_5, Regal.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
#p20.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p20.line(x55,y55_predicted,color='black',legend='y='+str(round(slope55,2))+'x+'+str(round(intercept55,2))+ '  ' + 'r^2 = ' + str(round(r_squared55,3)))

p20.legend.location='top_left'


p21 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Clarity Reference (ug/m^3)',
            y_axis_label='Wired 2 Calibrated PM 2.5 (ug/m3)',
            title = 'Clarity Reference vs Wired 2')

p21.circle(Paccar.Ref_PM2_5, Paccar.Predictions, size = 3, color = 'gray')#, legend = 'Paccar Adjusted Residuals')
#p21.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
p21.line(x66,y66_predicted,color='black',legend='y='+str(round(slope66,2))+'x+'+str(round(intercept66,2))+ '  ' + 'r^2 = ' + str(round(r_squared66,3)))

p21.legend.location='top_left'


p16 = gridplot([[p16,p17, p18], [p10, p11, p12]], plot_width = 400, plot_height = 300)
p22 = gridplot([[p19,p20, p21], [p13, p14, p15]], plot_width = 400, plot_height = 300)


tab3 = Panel(child=p16, title="Clarity Node vs Clarity Reference Comparisons")
tab4 = Panel(child=p22, title="Clarity Node vs Clarity Reference Comparisons")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4])


show(tabs)

export_png(p1, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_wired_time_series_mean_resample.png")
export_png(p9, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_scatter_mean_resample.png")
export_png(p16, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_residuals_and_predictions_1.png")    
export_png(p22, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_residuals_and_predictions_2.png")  
    
    
#%%

#def linear_plot(x,y,x_winter,y_winter,unit_name,n_lines,**kwargs):
linear_plot(Audubon.Ref_PM2_5, Audubon.Predictions, Audubon.Ref_PM2_5, Audubon.Predictions, 'Audubon', 1)
#%%
linear_plot(Balboa.Ref_PM2_5, Balboa.Predictions, Balboa.Ref_PM2_5, Balboa.Predictions, 'Balboa', 1)
#%%
linear_plot(Browne.Ref_PM2_5, Browne.Predictions, Browne.Ref_PM2_5, Browne.Predictions, 'Browne', 1)
#%%
linear_plot(Lidgerwood.Ref_PM2_5, Lidgerwood.Predictions, Lidgerwood.Ref_PM2_5, Lidgerwood.Predictions, 'Lidgerwood', 1)
#%%
linear_plot(Regal.Ref_PM2_5, Regal.Predictions, Regal.Ref_PM2_5, Regal.Predictions, 'Regal', 1)
    
    
    
    
    
    
    
    
    
    