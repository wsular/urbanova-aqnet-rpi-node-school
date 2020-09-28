#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 17:40:23 2020

@author: matthew
"""

import pandas as pd
from glob import glob
import numpy as np
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from scipy import stats
from bokeh.io import export_png
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure, show
from bokeh.models import Range1d, LinearAxis

PlotType = 'HTMLfile'


# Choose dates of interest
start_time = '2011-01-01 00:00'
#end_time = '2011-12-31 23:00'
end_time = '2014-12-31 23:00'

interval = '30T'
interval2 = 'M'
interval3 = 'D'

resample_dates = pd.read_excel (r'/Users/matthew/Desktop/506/Project/site_1_US/resample_dates_daily.xlsx')
dates = resample_dates['time']
test = list(resample_dates['test'])

resample_dates2 = pd.read_excel (r'/Users/matthew/Desktop/506/Project/site_1_US/resample_dates_daily2.xlsx')
test2 = list(resample_dates2['test2'])

resample_dates_monthly = pd.read_excel (r'/Users/matthew/Desktop/506/Project/site_1_US/resample_monthly_avg.xlsx')
resample_month = list(resample_dates_monthly['month'])

#%%
#Import entire data set

US_All = pd.read_excel (r'/Users/matthew/Desktop/506/Project/site_1_US/AMF_US-Jo2_BASE-BADM_1-5/AMF_US-Jo2_BASE_HH_1-5.xlsx')
US_All['time'] = pd.to_datetime(US_All['time'])
US_All = US_All.sort_values('time')
US_All.index = US_All.time
usa = US_All.loc[start_time:end_time]

usa = usa[usa['G_1_1_1'] > -9999]
usa = usa[usa['LE'] > -9999]
usa = usa[usa['NETRAD'] > -9999]
usa = usa[usa['WS'] > -9999]
usa = usa[usa['USTAR'] > -9999]
usa = usa[usa['RH'] > -9999]
usa = usa[usa['PA'] > -9999]
usa = usa[usa['VPD_PI'] > -9999]
usa = usa[usa['TA'] > -9999]
usa = usa[usa['LE'] > 10]
usa = usa[usa['H'] > 0]

usa['A'] = usa['NETRAD']-usa['G_1_1_1']  # Available energy in W/m^2
usa['Cp'] = 1006   # specific heat capacity of air (assumed constant) in J/Kg*K
usa['VPD_PI_units'] = usa['VPD_PI']*100   # change vapor pressure deficit units to Pa from hPa
usa['ra'] = usa['WS']/(usa['USTAR'])**2   # units s*m^-1 calculate aerodynamic resistance from mean wind speed and friction velocty (units sm^-1)

usa['delta_1'] = (17.27*usa['TA'])/(usa['TA']+237.3)          # exponential term of delta
usa['delta_2'] = 4096*(0.6108*(2.71828**(usa['delta_1'])))    # numerator of delta
usa['delta_3'] = (usa['TA'] + 237.3)**2                       # denominator of delta
usa['delta'] = usa['delta_2']/usa['delta_3']                  # delta (kPa/C) slope of saturation vapour pressure curve

usa['p_mb'] = usa['PA']*10       # converts pressure from kPa to mb for psychrometric constant calc
latent = 586     # latent heat of vaporization of water in cal/gram at 20 C
usa['psychrometric_1'] = (0.3861*usa['p_mb'])/latent
usa['psychrometric'] = usa['psychrometric_1']/10    # convert units from mb/C to kPa/C (same as kPa/K)

Rd = 287.058         # specific gas constant for dry air in J/(kg*K)
Rv = 461.495         # specific gas constant for water vapor in J/(kg*K)
usa['T_K'] = usa['TA']+273.15                                           # convert temperatures to Kelvin for saturation vapor pressure equation
usa['p1'] = 0.611*(2.71828**((17.625*usa['TA'])/(usa['TA']+243.04)))    # use Magnus calculate saturation vapor pressure (kPa)
usa['pv'] = usa['p1']*(usa['RH']/100)                                   # calculate actual vapor pressure (kPa)
usa['pd'] = ((usa['PA']) - usa['pv'])*1000                              # calculate pressure of dry air (factor of 1000 to convert pressure from kPa to Pa)
usa['density'] = ((usa['pd']/(Rd*usa['T_K'])) - ((usa['pv']*1000)/(Rv*usa['T_K'])))   # calculate air density in kg/m3  (factor of 1000 to convert kPa to Pa)


# calculation of rs
usa['rs_inner_left'] = (usa['delta']*usa['A'])/usa['LE']     # this factor ends up having units of kPa/K
usa['rs_inner_right'] = (usa['density']*usa['Cp']*usa['VPD_PI'])/(usa['ra']*usa['LE'])
usa['rs_inner_term'] = usa['rs_inner_left'] + usa['rs_inner_right'] - usa['delta'] - usa['psychrometric']
usa['rs_outer_term'] = usa['ra']/usa['psychrometric']
usa['rs'] = usa['rs_inner_term']*usa['rs_outer_term']    # surface resistance in units of s*m^-1


# calculation of sensitivity
usa['sensitivity'] = usa['delta']/(usa['delta']+usa['psychrometric']*(1+(usa['rs']/usa['ra'])))
usa['ratio'] = usa['rs']/usa['ra']
#%%
usa_resample_monthly = usa.resample(interval2).mean() 
#%%
usa_resample = usa.resample(interval3).mean() 
#%%
usa_resample1 = usa.groupby([usa.index.month]).mean()
usa_resample1['dates'] = resample_month
#%%
usa_resample2 = usa.groupby([usa.index.day,usa.index.month]).mean()
usa_resample2['dates'] = test
#%%
CA_All = pd.read_excel (r'/Users/matthew/Desktop/506/Project/site_2_CA/AMF_CA-Gro_BASE-BADM_2-5/AMF_CA-Gro_BASE_HH_2-5.xlsx')
CA_All['time'] = pd.to_datetime(CA_All['time'])
CA_All = CA_All.sort_values('time')
CA_All.index = CA_All.time
ca = CA_All.loc[start_time:end_time]

ca = ca[ca['G_1_1_1'] > -9999]
ca = ca[ca['LE'] > -9999]
ca = ca[ca['NETRAD_1_1_1'] > -9999]
ca = ca[ca['WS_1_1_1'] > -9999]
ca = ca[ca['USTAR'] > -9999]
ca = ca[ca['RH_1_1_1'] > -9999]
ca = ca[ca['PA'] > -9999]
ca = ca[ca['VPD_PI'] > -9999]
ca = ca[ca['TA_1_1_1'] > -9999]
ca = ca[ca['LE'] > 10]
ca = ca[ca['H'] > 0]

ca['A'] = ca['NETRAD_1_1_1']-ca['G_1_1_1']  # Available energy in W/m^2
ca['Cp'] = 1006   # specific heat capacity of air (assumed constant) in J/Kg*K
ca['VPD_PI_units'] = ca['VPD_PI']*100   # change vapor pressure deficit units to Pa from hPa
ca['ra'] = ca['WS_1_1_1']/(ca['USTAR'])**2   # calculate aerodynamic resistance from mean wind speed and friction velocty (units sm^-1)

ca['delta_1'] = (17.27*ca['TA_1_1_1'])/(ca['TA_1_1_1']+237.3)          # exponential term of delta
ca['delta_2'] = 4096*(0.6108*(2.71828**(ca['delta_1'])))    # numerator of delta
ca['delta_3'] = (ca['TA_1_1_1'] + 237.3)**2                       # denominator of delta
ca['delta'] = ca['delta_2']/ca['delta_3']                  # delta (kPa/C) slope of saturation vapour pressure curve

ca['p_mb'] = ca['PA']*10       # converts pressure from kPa to mb for psychrometric constant calc
latent = 586     # latent heat of vaporization of water in cal/gram at 20 C
ca['psychrometric_1'] = (0.3861*ca['p_mb'])/latent
ca['psychrometric'] = ca['psychrometric_1']/10    # convert units from mb/C to kPa/C (same as kPa/K)

Rd = 287.058         # specific gas constant for dry air in J/(kg*K)
Rv = 461.495         # specific gas constant for water vapor in J/(kg*K)
ca['T_K'] = ca['TA_1_1_1']+273.15                                           # convert temperatures to Kelvin for saturation vapor pressure equation
ca['p1'] = 0.611*(2.71828**((17.625*ca['TA_1_1_1'])/(ca['TA_1_1_1']+243.04)))    # use Magnus calculate saturation vapor pressure (kPa)
ca['pv'] = ca['p1']*(ca['RH_1_1_1']/100)                                   # calculate actual vapor pressure (kPa)
ca['pd'] = ((ca['PA']) - ca['pv'])*1000                              # calculate pressure of dry air (factor of 1000 to convert pressure from kPa to Pa)
ca['density'] = ((ca['pd']/(Rd*ca['T_K'])) - ((ca['pv']*1000)/(Rv*ca['T_K'])))   # calculate air density in kg/m3  (factor of 1000 to convert kPa to Pa)

# calculation of rs
ca['rs_inner_left'] = (ca['delta']*ca['A'])/ca['LE']     # this factor ends up having units of kPa/K
ca['rs_inner_right'] = (ca['density']*ca['Cp']*ca['VPD_PI'])/(ca['ra']*ca['LE'])
ca['rs_inner_term'] = ca['rs_inner_left'] + ca['rs_inner_right'] - ca['delta'] - ca['psychrometric']
ca['rs_outer_term'] = ca['ra']/ca['psychrometric']
ca['rs'] = ca['rs_inner_term']*ca['rs_outer_term']    # surface resistance in units of s*m^-1


# calculation of sensitivity
ca['sensitivity'] = ca['delta']/(ca['delta']+ca['psychrometric']*(1+(ca['rs']/ca['ra'])))
ca['ratio'] = ca['rs']/ca['ra']
#%%
ca_resample_monthly = ca.resample(interval2).mean() 
#%%
ca_resample = ca.resample(interval3).mean() 
#%%
ca_resample1 = ca.groupby([ca.index.month]).mean()
ca_resample1['dates'] = resample_month
#%%
ca_resample2 = ca.groupby([ca.index.day,ca.index.month]).mean()
ca_resample2['dates'] = test2
#%%
# Plotting

#if PlotType=='notebook':
#    output_notebook()
#else:
#    output_file('/Users/matthew/Desktop/clarity_RH_time_series_legend_hide.html')


# Monthly resample of sensitivity and latent heat

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='Sensitivity')

p1.title.text = 'Sensitivity'        
p1.y_range = Range1d(start=0, end=0.08)
p1.line(usa_resample_monthly.index,     usa_resample_monthly.sensitivity,     legend='Shrubland sensitivity',       color='green',            line_width=2)
p1.line(ca_resample_monthly.index,      ca_resample_monthly.sensitivity,      legend='Boreal Forest sensitivity',        color='blue',             line_width=2)

p1.extra_y_ranges['LE'] = Range1d(start=0, end=150)
p1.add_layout(LinearAxis(y_range_name='LE', axis_label='LE [W/m^2]'), 'right')


p1.line(usa_resample_monthly.index,     usa_resample_monthly.LE,   y_range_name = 'LE',  legend='Shrubland LE',       color='black',            line_width=2)
p1.line(ca_resample_monthly.index,      ca_resample_monthly.LE,    y_range_name = 'LE',  legend='Boreal Forest LE',        color='red',             line_width=2)

p1.legend.click_policy="hide"

# Monthly resample of resistances

p11 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='rs (sm^-1')

p11.title.text = 'Resistance'        
p11.y_range = Range1d(start=0, end=40000)
p11.line(usa_resample_monthly.index,     usa_resample_monthly.rs,     legend='USA rs',       color='green',            line_width=2)
p11.line(ca_resample_monthly.index,      ca_resample_monthly.rs,      legend='CA rs',        color='blue',             line_width=2)

p11.extra_y_ranges['ra'] = Range1d(start=0, end=1000)
p11.add_layout(LinearAxis(y_range_name='ra', axis_label='ra [sm^-1]'), 'right')


p11.line(usa_resample_monthly.index,     usa_resample_monthly.ra,   y_range_name = 'ra',  legend='USA ra',       color='black',            line_width=2)
p11.line(ca_resample_monthly.index,      ca_resample_monthly.ra,    y_range_name = 'ra',  legend='CA ra',        color='red',             line_width=2)

p11.legend.click_policy="hide"


p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='Ratio')

p2.title.text = 'Resistance Ratios'        

#p2.line(usa_resample.index,     usa_resample.LE,     legend='USA',       color='green',            line_width=2)
#p2.line(ca_resample.index,      ca_resample.LE,      legend='CA',        color='blue',             line_width=2)

p2.line(usa_resample.index,     usa_resample.ratio,     legend='Shrubland ratio',       color='black',            line_width=2)
p2.line(ca_resample.index,      ca_resample.ratio,      legend='Boreal Forest ratio',        color='red',             line_width=2)


p2.legend.click_policy="hide"


p3 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='LE (W/m^2)')

p3.title.text = 'Daily Latent Heat'        

p3.line(usa_resample.index,     usa_resample.LE,     legend='USA',       color='green',            line_width=2)
p3.line(ca_resample.index,      ca_resample.LE,      legend='CA',        color='blue',             line_width=2)



p3.legend.click_policy="hide"

p4 = figure(plot_width=900,
            plot_height=450,
           # x_axis_type='datetime',
            x_axis_label='Month',
            y_axis_label='Resistance (s/m')

p4.title.text = 'Resistances'        

p4.line(usa_resample1.dates,     usa_resample1.rs,     legend='USA rs',       color='green',            line_width=2)
p4.line(ca_resample1.dates,      ca_resample1.rs,      legend='CA rs',        color='blue',             line_width=2)
p4.line(usa_resample1.dates,     usa_resample1.ra,     legend='USA ra',       color='black',            line_width=2)
p4.line(ca_resample1.dates,      ca_resample1.ra,      legend='CA ra',        color='red',             line_width=2)


p4.legend.click_policy="hide"


p5 = figure(plot_width=900,
            plot_height=450,
           # x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='Resistance (s/m')

p5.title.text = 'Resistances'        

p5.line(usa_resample2.dates,     usa_resample2.rs,     legend='USA rs',       color='green',            line_width=2)
p5.line(ca_resample2.dates,      ca_resample2.rs,      legend='CA rs',        color='blue',             line_width=2)
p5.line(usa_resample2.dates,     usa_resample2.ra,     legend='USA ra',       color='black',            line_width=2)
p5.line(ca_resample2.dates,      ca_resample2.ra,      legend='CA ra',        color='red',             line_width=2)
p5.line(usa_resample2.dates,     usa_resample2.LE,     legend='USA LE',       color='grey',            line_width=2)
p5.line(ca_resample2.dates,      ca_resample2.LE,      legend='CA LE',        color='teal',             line_width=2)


p5.legend.click_policy="hide"

p6 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='Resistance (s/m')

p6.title.text = 'Resistances'        

p6.line(usa.index,     usa.LE,     legend='USA LE',       color='green',            line_width=2)
p6.line(ca.index,      ca.LE,      legend='CA LE',        color='blue',             line_width=2)
p6.line(usa.index,     usa.A,     legend='USA A',       color='black',            line_width=2)
p6.line(ca.index,      ca.A,      legend='CA A',        color='red',             line_width=2)
p6.line(usa.index,     usa.NETRAD,     legend='USA NETRAD',       color='purple',            line_width=2)
p6.line(ca.index,      ca.NETRAD_1_1_1,      legend='CA NETRAD',        color='brown',             line_width=2)
p6.line(usa.index,     usa.rs,     legend='USA rs',       color='grey',            line_width=2)
p6.line(ca.index,      ca.rs,      legend='CA rs',        color='teal',             line_width=2)
p6.line(usa.index,     usa.ra,     legend='USA ra',       color='orange',            line_width=2)
p6.line(ca.index,      ca.ra,      legend='CA ra',        color='yellow',             line_width=2)

p6.legend.click_policy="hide"

# plot of sensible heat

p7 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='H (W/m^2)')

p7.title.text = 'Daily Sensible Heat'        

p7.line(usa_resample.index,     usa_resample.H,     legend='Shrubland H',       color='green',            line_width=2)
p7.line(ca_resample.index,      ca_resample.H,      legend='Boreal H',        color='blue',             line_width=2)



tab1 = Panel(child=p1, title="Sensitivity Monthly Resample")
tab11 = Panel(child=p11, title="Monthly Averaged Resistances")
tab2 = Panel(child=p2, title='Latent Heat Daily Resample and Ratios')
tab3 = Panel(child=p3, title='Latent Heat All Daily Resample')
tab4 = Panel(child=p4, title='Resistances Monthly all 4 yrs Averaged')
tab5 = Panel(child=p5, title='Resistances Daily Averages all 4 yrs avergaed')
tab6 = Panel(child=p6, title='All data')
tab7 = Panel(child=p7, title='Sensible Heat')

tabs = Tabs(tabs=[ tab1,tab11, tab2, tab7, tab3, tab4, tab5, tab6])

show(tabs)



