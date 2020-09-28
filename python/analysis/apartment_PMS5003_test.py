#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 09:22:16 2020

@author: matthew
"""

# prototype time series from apartment data

PlotType = 'HTMLfile'
from bokeh.io import export_png
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from glob import glob
from bokeh.models import Span, Label
import pandas as pd


interval = '2T'

#Apartment Test

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/Apartment/apartment_time_series.html')
    #output_file('/Users/matthew/Desktop/Apartment/apartment_time_series_log_y.html')
    
proto = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/Apartment/prototype/apartment/WSU*.csv')
files.sort()
for file in files:
    proto = pd.concat([proto, pd.read_csv(file)], sort=False)
    
proto['DateTime'] = pd.to_datetime(proto['DateTime'])
proto = proto.sort_values('DateTime')
proto.index = proto.DateTime

sensor7 = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/Apartment/Sensor_7_Good_Data/apartment/WSU*.csv')
files.sort()
for file in files:
    sensor7 = pd.concat([sensor7, pd.read_csv(file)], sort=False)
    
sensor7['DateTime'] = pd.to_datetime(sensor7['DateTime'])
sensor7 = sensor7.sort_values('DateTime')
sensor7.index = sensor7.DateTime

DustTrak = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/Apartment/Dust_Trak/*.csv')
files.sort()
for file in files:
    DustTrak = pd.concat([DustTrak, pd.read_csv(file)], sort=False)
    
    
DustTrak['DateTime'] = pd.to_datetime(DustTrak['DateTime'])
DustTrak = DustTrak.sort_values('DateTime')
DustTrak.index = DustTrak.DateTime

proto = proto.resample(interval).mean()  
sensor7 = sensor7.resample(interval).mean()
DustTrak = DustTrak.resample(interval).mean()

#%%

start =pd.to_datetime('2020-01-30T17:35:00Z')
cook2 = pd.to_datetime('2020-01-31T19:45:00Z')
cook3 = pd.to_datetime('2020-02-01T20:00:00Z')
cook4 = pd.to_datetime('2020-02-02T11:00:00Z')
cook5 = pd.to_datetime('2020-02-02T14:00:00Z')
cook6 = pd.to_datetime('2020-02-02T16:14:00Z')
cook7 = pd.to_datetime('2020-02-02T20:30:00Z')
cook8 = pd.to_datetime('2020-01-31T07:00:00Z')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
           # y_axis_type='log',
            y_axis_label='PM 2.5 (ug/m^3)')
            
p1.line(proto.index,      proto.PM2_5_standard,              legend='Prototype',      color='blue',       line_width=2)
p1.line(sensor7.index,    sensor7.PM2_5_standard,            legend='Sensor 7',       color='green',      line_width=2)
p1.line(DustTrak.index,   DustTrak.PM2_5,                    legend='Dust Trak',      color='red',        line_width=2)

cook_span = Span(location = start, dimension='height', line_color = 'gray', line_width = 3)
p1.renderers.extend([cook_span])
cook_label = Label(x = start, y=60,text='Middle of Cooking', text_font_size = '8pt')
p1.add_layout(cook_label)


cook_span2 = Span(location = cook2, dimension='height', line_color = 'gray', line_width = 3)
p1.renderers.extend([cook_span2])
cook_label = Label(x = cook2, y=400,text='Start Cooking', text_font_size = '8pt')
p1.add_layout(cook_label)

cook_span3 = Span(location = cook3, dimension='height', line_color = 'gray', line_width = 3)
p1.renderers.extend([cook_span3])
cook_label = Label(x = cook3, y=400,text='Start Cooking Lamb', text_font_size = '8pt')
p1.add_layout(cook_label)

cook_span4 = Span(location = cook4, dimension='height', line_color = 'gray', line_width = 3)
p1.renderers.extend([cook_span4])
cook_label = Label(x = cook4, y=850,text='Start Cooking', text_font_size = '8pt')
cook_label2 = Label(x = cook4, y=800,text='Stir Fry, Chicken', text_font_size = '8pt')
p1.add_layout(cook_label)
p1.add_layout(cook_label2)

cook_span5 = Span(location = cook5, dimension='height', line_color = 'gray', line_width = 3)
p1.renderers.extend([cook_span5])
cook_label = Label(x = cook5, y=500,text='Start Cooking', text_font_size = '8pt')
p1.add_layout(cook_label)

cook_span6 = Span(location = cook6, dimension='height', line_color = 'gray', line_width = 3)
#p1.renderers.extend([cook_span6])
cook_label = Label(x = cook6, y=600,text='Smoke Alarm', text_font_size = '8pt')
p1.add_layout(cook_label)

cook_span7 = Span(location = cook7, dimension='height', line_color = 'gray', line_width = 3)
#p1.renderers.extend([cook_span6])
cook_label = Label(x = cook7, y=100,text='Boiling Beans', text_font_size = '8pt')
p1.add_layout(cook_label)

cook_span8 = Span(location = cook8, dimension='height', line_color = 'gray', line_width = 3)
#p1.renderers.extend([cook_span6])
cook_label = Label(x = cook8, y=20,text='Scrambled Eggs', text_font_size = '8pt')
p1.add_layout(cook_label)



p1.legend.location='top_left'

tab1 = Panel(child=p1, title="Prototype PM 2.5")
tabs = Tabs(tabs=[ tab1 ]) 
show(tabs)

export_png(p1, filename="/Users/matthew/Desktop/Apartment/apartment_time_series.png")
#export_png(p1, filename="/Users/matthew/Desktop/Apartment/apartment_time_series_log_y.png")
#%%

# Scatter of Prototype and Sensor 7
# Resampled to 1 minute so data match up

PlotType = 'HTMLfile'
from bokeh.io import export_png
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from glob import glob
from bokeh.models import Span, Label
import pandas as pd
import numpy as np
import scipy 
from bokeh.layouts import row
from bokeh.layouts import gridplot


interval = '1T'


#Apartment Test

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/Apartment/apartment_scatter.html')

proto = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/Apartment/prototype/apartment/WSU*.csv')
files.sort()
for file in files:
    proto = pd.concat([proto, pd.read_csv(file)], sort=False)
    
proto['Datetime'] = pd.to_datetime(proto['Datetime'])
proto = proto.sort_values('Datetime')
proto.index = proto.Datetime
#proto = proto.loc[start_time:end_time]
proto = proto.resample(interval).mean()


sensor7 = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/Apartment/Sensor_7_Good_Data/apartment/WSU*.csv')
files.sort()
for file in files:
    sensor7 = pd.concat([sensor7, pd.read_csv(file)], sort=False)
    
sensor7['DateTime'] = pd.to_datetime(sensor7['DateTime'])
sensor7 = sensor7.sort_values('DateTime')
sensor7.index = sensor7.DateTime
#sensor7 = sensor7.loc[start_time:end_time]
sensor7 = sensor7.resample(interval).mean()

# tab for plotting scatter plots of clarity nodes vs clarity "Reference" wired node
df = pd.DataFrame()
df['proto'] = proto['PM2_5_standard']
df['sensor7'] = sensor7['PM2_5_standard']

df = df.dropna()

#the data for proto 1 to 1 line
x=np.array(df.proto)
y=np.array(df.proto)

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

#the data for proto vs sensor7
x1=np.array(df.proto)
y1=np.array(df.sensor7) 
slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x1, y1)
r_squared1 = r_value11**2
# determine best fit line
par = np.polyfit(x1, y1, 1, full=True)
slope1=par[0][0]
intercept1=par[0][1]
y1_predicted = [slope1*i + intercept1  for i in x1]


p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='PMS5003 Prototype (ug/m^3)',
            y_axis_label='PMS5003 Sensor7 (ug/m^3)')

#p2.circle(x,y,legend='Reference 1 to 1 line', color='red')
p2.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

p2.circle(x1,y1,legend='Proto vs Sensor7', color='blue')
p2.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

p2.legend.location='top_left'
p2.toolbar.logo = None
p2.toolbar_location = None

tab2 = Panel(child=p2, title="PMS5003 Indoor Apt. Comparison")

tabs = Tabs(tabs=[ tab2])


show(tabs)

export_png(p2, filename="/Users/matthew/Desktop/apartment/PMS5003_apartment_mean_resample_scatter.png")

#%%

### Histograms of PM 2.5 measurement distributions

import numpy as np
import holoviews as hv

hv.extension('bokeh', logo=False)

data = sensor7['PM2_5_standard']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p2 = figure(plot_width = 1500,
            plot_height = 700)

p2 = hv.Histogram((edges, frequencies))
p2 = p2.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'Sensor 7')

data = proto['PM2_5_standard']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p3 = figure(plot_width = 1500,
            plot_height = 700)

p3 = hv.Histogram((edges, frequencies))
p3 = p3.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'Prototype')

p4 = (p2+p3).cols(2)

hv.save(p4.options(toolbar=None), '/Users/matthew/Desktop/Apartment/histogram_distributions.png' , fmt='png', backend='bokeh')    # works

show(hv.render(p4))



















