#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:35:17 2020

@author: matthew
"""


from bokeh.models import Panel, Tabs
from bokeh.io import show
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure
from bokeh.plotting import reset_output


def indoor_outdoor_plot(indoor, outdoor):

    location = indoor.iloc[0]['Location']
    
    p1 = figure(title = location,
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
    p1.line(indoor.index,     indoor.PM2_5_corrected,  legend='Indoor PMS5003 Calibrated',  muted_color='red', muted_alpha=0.3,     color='red',     line_width=2)
    p1.line(outdoor.index,     outdoor.PM2_5_corrected,   legend='Outdoor Clarity Calibrated',  muted_color='black', alpha=0.2,     color='black',       line_width=2)
   # p1.line(indoor.index,     indoor.out_PM2_5_corrected,   legend='Outdoor Clarity Calibrated',  muted_color='black', alpha=0.2,     color='black',       line_width=2) 
    p1.line(indoor.index,     indoor.PM2_5_corrected_shift,  legend='Shifted',  muted_color='blue', alpha=0.4,     color='blue',     line_width=2)

    
    source_error = ColumnDataSource(data=dict(base=outdoor.index, lower=outdoor.lower_uncertainty, upper=outdoor.upper_uncertainty))

    #p1.add_layout(
    #    Whisker(source=source_error, base="base", upper="upper", lower="lower")
    #)

   # source_error_indoor = ColumnDataSource(data=dict(base=indoor.index, lower=indoor.lower_uncertainty, upper=indoor.upper_uncertainty))

    #p1.add_layout(
    #    Whisker(source=source_error_indoor, base="base", upper="upper", lower="lower")
    #)

    p1.legend.click_policy="mute"
    p1.legend.location='top_left'
        
    tab1 = Panel(child=p1, title="Indoor Outdoor Comparison")
    
    tabs = Tabs(tabs=[ tab1])
    
    show(tabs)

    reset_output()