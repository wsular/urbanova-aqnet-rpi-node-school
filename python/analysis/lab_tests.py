#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 14:03:37 2019

@author: matthew
"""

#%%
PlotType = 'HTMLfile'

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from glob import glob
from bokeh.models import Span, Label

# Lab solder smoke test INDOOR SENSOR 7
#11/13 test WSU_LAR_Indoor_Air_Quality_Node_7_20191113*

import pandas as pd
#%%
lab_test = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/lab_tests/11_20_19/WSU_LAR_Indoor_Air_Quality_Node_7_20191120_155013.json')
files.sort()
for file in files:
    lab_test = pd.concat([lab_test, pd.read_json(file)], sort=False)
lab_test.index = lab_test.Datetime

#%%
#Add in start/stop and plume times for 11/20
plume = [pd.to_datetime('2019-11-20T16:00:00Z'), 
         pd.to_datetime('2019-11-20T16:20:00Z'),
         pd.to_datetime('2019-11-20T16:40:00Z'),
         ]

#%%
# Add in start/stop data acquisition and plume times for 11/13

start =[pd.to_datetime('2019-11-13T14:10:00Z'), 
        pd.to_datetime('2019-11-13T14:30:00Z'),
        pd.to_datetime('2019-11-13T14:50:00Z'),
        pd.to_datetime('2019-11-13T15:10:00Z'),
        pd.to_datetime('2019-11-13T15:50:00Z'),
        pd.to_datetime('2019-11-13T16:30:00Z'),
       ]

stop = [pd.to_datetime('2019-11-13T14:20:00Z'), 
        pd.to_datetime('2019-11-13T14:40:00Z'),
        pd.to_datetime('2019-11-13T15:00:00Z'),
        pd.to_datetime('2019-11-13T15:40:00Z'),
        pd.to_datetime('2019-11-13T16:20:00Z'),
        pd.to_datetime('2019-11-13T17:00:00Z'),
       ]

plume = [pd.to_datetime('2019-11-13T15:20:00Z'), 
         pd.to_datetime('2019-11-13T15:30:00Z'),
         pd.to_datetime('2019-11-13T16:00:00Z'),
         pd.to_datetime('2019-11-13T16:10:00Z'),
         pd.to_datetime('2019-11-13T16:40:00Z'),
         pd.to_datetime('2019-11-13T16:50:00Z'),
         ]


#%%
#Plot lab test time series

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/lab_tests/11_13_19/test_time_series.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m^3)')
            
p1.line(lab_test.index,      lab_test.PM2_5_standard,              legend='Lab Test',      color='blue',       line_width=2)


#for time in start:
#    start_span = Span(location = time, dimension='height', line_color = 'green', line_width = 2)
#    p1.renderers.extend([start_span])
#    start_label = Label(x=time, y=2000,text='Start ', text_font_size = '8pt')
#    p1.add_layout(start_label)


#for time in stop:
#    stop_span = Span(location = time, dimension='height', line_color = 'red', line_width = 2)
#    p1.renderers.extend([stop_span])
#    stop_label = Label(x=time, y=1500,text='Stop ', text_font_size = '8pt')
#    p1.add_layout(stop_label)
    
for time in plume:
    plume_span = Span(location = time, dimension='height', line_color = 'gray', line_width = 3)
    p1.renderers.extend([plume_span])
    plume_label = Label(x=time, y=1000)#,text='Plume Created', text_font_size = '8pt')
    p1.add_layout(plume_label)

p1.legend.location='top_left'

tab1 = Panel(child=p1, title="Sensor 7 PM 2.5")
tabs = Tabs(tabs=[ tab1 ]) 
show(tabs)

#%%

# Reset Test

reset_test = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/lab_tests/reset_test/Reset*.json')
files.sort()

for file in files:
    reset_test = pd.concat([reset_test, pd.read_json(file)], sort=False)
reset_test.index = reset_test.Datetime


if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/lab_tests/reset_test/no_reset_test_time_series.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m^3)')
            
p1.line(reset_test.index,      reset_test.PM2_5_standard,              legend='Lab Reset Test',      color='blue',       line_width=2)
p1.legend.location='top_left'

tab1 = Panel(child=p1, title="Sensor 7 PM 2.5")
tabs = Tabs(tabs=[ tab1 ]) 
show(tabs)

#%%
PlotType = 'HTMLfile'

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from glob import glob
from bokeh.models import Span, Label
import pandas as pd


#Apartment Test

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/Sensor_7_Good_Data/apartment/apartment_time_series.html')

apt_test = pd.DataFrame({})
    
files = glob('/Users/matthew/Desktop/Sensor_7_Good_Data/apartment/WSU*.json')
files.sort()
for file in files:
    apt_test = pd.concat([apt_test, pd.read_json(file)], sort=False)
apt_test.index = apt_test.Datetime


p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m^3)')
            
p1.line(apt_test.index,      apt_test.PM2_5_standard,              legend='Apartment Test',      color='blue',       line_width=2)
p1.legend.location='top_left'

tab1 = Panel(child=p1, title="Sensor 7 PM 2.5")
tabs = Tabs(tabs=[ tab1 ]) 
show(tabs)






