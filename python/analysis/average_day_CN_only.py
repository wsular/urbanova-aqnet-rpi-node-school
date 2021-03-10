#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:17:30 2021

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:38:06 2021

@author: matthew
"""

from bokeh.plotting import figure
from bokeh.models import Panel, Tabs

import pandas as pd
from glob import glob
from figure_format import figure_format
from bokeh.models.formatters import DatetimeTickFormatter

# don't really need the shift parameter as only concerned with the original corrected data, the shift was just to estimate lag-times, not for any other calcs
def average_day_CN(outdoor, site_number, time_period, smoke):
    
     # file path based on the time period
  #  if time_period == '6':
  #     # print('1')
  #      filepath = '/Users/matthew/Desktop/thesis/Final_Figures/CN_only/All_without_smoke/hourly'
   
    if time_period == '6' and smoke =='yes':
        y_scale_option = (5, 11.5)
    elif time_period == '6' and smoke == 'no':
        y_scale_option = (1.8,9)
    elif time_period == '7':
        y_scale_option = (5,31.5)
    
    
    dates = pd.DataFrame({})

    files = glob('/Users/matthew/Desktop/daily_test/dummy_date*.csv')
    files.sort()
    for file in files:
        dates = pd.concat([dates, pd.read_csv(file)], sort=False)

  
    outdoor_average_day = pd.DataFrame({})

    
    outdoor_average_day['PM2_5_hourly_avg'] = outdoor['PM2_5_corrected'].groupby(outdoor.index.hour).mean()
    outdoor_average_day['times'] = pd.to_datetime(dates['times'])
    outdoor_average_day = outdoor_average_day.sort_values('times')
    outdoor_average_day.index = outdoor_average_day.times
    
    averages = pd.DataFrame({})
    averages[outdoor.iloc[0]['Location'] + '_CN'] = outdoor_average_day.PM2_5_hourly_avg.round(2)
    print(averages)
    #print('outdoor', (outdoor_average_day.PM2_5_hourly_avg).round(2))
    #print('indoor' , (indoor_average_day.PM2_5_hourly_avg).round(2))
        

    p1 = figure(plot_width=900,
                        plot_height=450,
                        x_axis_type='datetime',
                        x_axis_label='Time (hrs)',
                        y_axis_label='PM2.5 (ug/m^3)',
                        y_range = y_scale_option)
    
    p1.title.text = site_number
    p1.title.text_font_size = '14pt'

            
    p1.circle(outdoor_average_day.index,       outdoor_average_day.PM2_5_hourly_avg,    size = 8,   legend='CN',        color='black',              line_width=2, muted_color='blue', muted_alpha=0.2)
    p1.line(outdoor_average_day.index,       outdoor_average_day.PM2_5_hourly_avg,              color='black',              line_width=2, muted_color='blue', muted_alpha=0.2)
    
    
    p1.legend.click_policy="mute"
    figure_format(p1)
    p1.legend.location='top_left'
    p1.xaxis.formatter = DatetimeTickFormatter(days="", hours="%H", seconds="" )
        
    
    tab1 = Panel(child=p1, title="Average Hour Values")
    
    tabs = Tabs(tabs=[ tab1])
    
   # show(tabs) 
    
    return(p1, averages)