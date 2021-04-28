#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 15:26:31 2021

@author: matthew
"""
from glob import glob
import pandas as pd
import numpy as np
import scipy
from bokeh.plotting import figure
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
import shapely
from shapely.geometry import LineString, Point
from bokeh.models import Label, ranges
import statsmodels.api as sm
from bokeh.io import export_png, output_file
from bokeh.layouts import gridplot

#%%

start_time = '2019-10-31 07:00'
end_time = '2019-11-08 19:00'

Audubon_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Audubon*.csv')
files.sort()
for file in files:
    Audubon_All = pd.concat([Audubon_All, pd.read_csv(file)], sort=False)
    
Audubon_All['time'] = pd.to_datetime(Audubon_All['time'])
Audubon_All = Audubon_All.sort_values('time')
Audubon_All.index = Audubon_All.time
Audubon = Audubon_All.loc[start_time:end_time]
#%%

    
p1 = figure(title = 'Site #6',
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m³)',
            y_range= ranges.Range1d(start=0,end=60)) 
p1.title.text_font_size = '14pt' 

p1.line(Audubon.index,     Audubon.PM2_5,              muted_color='red', muted_alpha=0.4,   color='black', line_alpha=0.4,      line_width=2) # legend='Calibrated CN',


p1.xaxis.axis_label_text_font_size = "14pt"
p1.xaxis.major_label_text_font_size = "14pt"
p1.xaxis.axis_label_text_font = "times"
p1.xaxis.axis_label_text_color = "black"
p1.xaxis.major_label_text_font = "times"

p1.yaxis.axis_label_text_font_size = "14pt"
p1.yaxis.major_label_text_font_size = "14pt"
p1.yaxis.axis_label_text_font = "times"
p1.yaxis.axis_label_text_color = "black"
p1.yaxis.major_label_text_font = "times"

p1.toolbar.logo = None
p1.toolbar_location = None
    
p1.xgrid.grid_line_color = None
p1.ygrid.grid_line_color = None
    
#p1.x_range.range_padding = 0
#p1.y_range.range_padding = 0
p2 = gridplot([[p1]], plot_width = 450, plot_height = 300, toolbar_location=None)
#export_png(p2, filename='/Users/matthew/Desktop/cal_gridplot_compare_y_axis.png')

tab1 = Panel(child=p1, title="Indoor Outdoor Comparison")

tabs = Tabs(tabs=[ tab1])

show(tabs)