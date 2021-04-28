#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:38:06 2021

@author: matthew
"""

import numpy as np
import scipy
from bokeh.plotting import figure
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
import shapely
from shapely.geometry import LineString, Point
from bokeh.models import Label
import statsmodels.api as sm
from bokeh.io import export_png, output_file
import pandas as pd
from glob import glob
from figure_format import figure_format
from bokeh.models.formatters import DatetimeTickFormatter

# don't really need the shift parameter as only concerned with the original corrected data, the shift was just to estimate lag-times, not for any other calcs
def average_day(indoor, outdoor, site_number, time_period, shift):
    
     # file path based on the time period
    if time_period == '1':
       # print('1')
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_1/'
        y_scale_option = (-0.5, 10)
    elif time_period == '2':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_2/'
        y_scale_option = (-0.5, 7)
    elif time_period == '3':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_3/'
        y_scale_option = (0, 250)
    elif time_period == '4':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_4/'
        y_scale_option = (-0.5, 10)
    elif time_period == '5':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_5/'
        y_scale_option = (0, 90)
    
    
    PlotType = 'HTMLfile'
    
    dates = pd.DataFrame({})

    files = glob('/Users/matthew/Desktop/daily_test/dummy_date*.csv')
    files.sort()
    for file in files:
        dates = pd.concat([dates, pd.read_csv(file)], sort=False)

    indoor_average_day = pd.DataFrame({})
    outdoor_average_day = pd.DataFrame({})
    # use for unshifted corrected data
    if shift == 'unshifted':
        indoor_average_day['PM2_5_hourly_avg'] = indoor['PM2_5_corrected'].groupby(indoor.index.hour).mean()
    # use for shifted corrected data - don't actually need this
   # elif shift == 'shifted':
   #     indoor_average_day['PM2_5_hourly_avg'] = indoor['PM2_5_corrected_shift'].groupby(indoor.index.hour).mean()
        
    indoor_average_day['times'] = pd.to_datetime(dates['times'])
    indoor_average_day = indoor_average_day.sort_values('times')
    indoor_average_day.index = indoor_average_day.times
    
    
    outdoor_average_day['PM2_5_hourly_avg'] = outdoor['PM2_5_corrected'].groupby(outdoor.index.hour).mean()
    outdoor_average_day['times'] = pd.to_datetime(dates['times'])
    outdoor_average_day = outdoor_average_day.sort_values('times')
    outdoor_average_day.index = outdoor_average_day.times
    
    averages = pd.DataFrame({})
    averages[indoor.iloc[0]['Location'] + '_IAQU'] = indoor_average_day.PM2_5_hourly_avg.round(2)
    averages[indoor.iloc[0]['Location'] + '_CN'] = outdoor_average_day.PM2_5_hourly_avg.round(2)
    print(averages)
    #print('outdoor', (outdoor_average_day.PM2_5_hourly_avg).round(2))
    #print('indoor' , (indoor_average_day.PM2_5_hourly_avg).round(2))
        
    if PlotType=='notebook':
        output_notebook()
    else:
        output_file('/Users/matthew/Desktop/clarity_PM2.5_time_series_legend_mute.html')
            
    p1 = figure(plot_width=900,
                        plot_height=450,
                        x_axis_type='datetime',
                        x_axis_label='Time (hrs)',
                        y_axis_label='PM2.5 (ug/m³)',
                        y_range = y_scale_option)
    p1.title.text = site_number
    p1.title.text_font_size = '14pt'
    p1.title.text_font = 'times'
            
    p1.triangle(indoor_average_day.index,     indoor_average_day.PM2_5_hourly_avg, size = 8,             color='black',             line_width=2, muted_color='black', muted_alpha=0.2)
    p1.line(indoor_average_day.index,     indoor_average_day.PM2_5_hourly_avg,             color='black',             line_width=2, muted_color='black', muted_alpha=0.2)
    p1.circle(outdoor_average_day.index,       outdoor_average_day.PM2_5_hourly_avg,    size = 8,          color='black',              line_width=2, muted_color='blue', muted_alpha=0.2)
    p1.line(outdoor_average_day.index,       outdoor_average_day.PM2_5_hourly_avg,              color='black',              line_width=2, muted_color='blue', muted_alpha=0.2)
    
    
    p1.legend.click_policy="mute"
    figure_format(p1)
    p1.legend.location='top_center'
    p1.xaxis.formatter = DatetimeTickFormatter(days="", hours="%H", seconds="" )
    
    p1.yaxis.major_label_text_font = "times"
    p1.xaxis.major_label_text_font = "times"
    
    if shift == 'unshifted':
        export_png(p1, filename=filepath + 'hourly_averages_' + indoor.iloc[0]['Location'] + '.png')

    
    
    tab1 = Panel(child=p1, title="Average Hour Values")
    
    tabs = Tabs(tabs=[ tab1])
    
   # show(tabs) 
    
    return(p1, averages)