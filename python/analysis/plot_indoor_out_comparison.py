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
from bokeh.io import export_png, output_file


def indoor_outdoor_plot(indoor, outdoor, site_number, time_period):

    # file path based on the time period
    if time_period == '1':
       # print('1')
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_1/'
    elif time_period == '2':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_2/'
    elif time_period == '3':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_3/'
    elif time_period == '4':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_4/'
    
    # use if want to label site by school name
    #location = indoor.iloc[0]['Location']
    
    # use if want to label site by number (numbered in alphabetical order)
    location = site_number
    
    p1 = figure(
            #title = location,
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
    # just plotting the calibrated shifted indoor pm2.5 so not so cluttered, unless the shift doesn't make any sense and need a visual to explain
    #p1.line(indoor.index,     indoor.PM2_5_corrected,  legend='Calibrated IAQU',  muted_color='black', muted_alpha=0.3,     color='black',     line_width=2)
    p1.line(outdoor.index,     outdoor.PM2_5_corrected,   legend='Calibrated CN',  muted_color='red', alpha=0.4,     color='red',       line_width=2)
   # p1.line(indoor.index,     indoor.out_PM2_5_corrected,   legend='Outdoor Clarity Calibrated',  muted_color='black', alpha=0.2,     color='black',       line_width=2) 
    p1.line(indoor.index,     indoor.PM2_5_corrected_shift,  legend='Calibrated IAQU',  muted_color='black',     color='black',  line_width=2)#, alpha=0.5)

    
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
    
    # remove bokeh and logo from final pngs (keep in so can look at data before final saved pic)
    p1.toolbar.logo = None
    p1.toolbar_location = None
    p1.xgrid.grid_line_color = None
    p1.ygrid.grid_line_color = None
    
    p1.legend.location='top_left'
    p1.legend.label_text_font_size = "14pt"
    p1.legend.label_text_font = "times"
    p1.legend.label_text_color = "black"
    
   # p1.xaxis.axis_label="xaxis_name"
    p1.xaxis.axis_label_text_font_size = "14pt"
    p1.xaxis.major_label_text_font_size = "14pt"
    p1.xaxis.axis_label_text_font = "times"
    p1.xaxis.axis_label_text_color = "black"

   # p1.yaxis.axis_label="yaxis_name"
    p1.yaxis.axis_label_text_font_size = "14pt"
    p1.yaxis.major_label_text_font_size = "14pt"
    p1.yaxis.axis_label_text_font = "times"
    p1.yaxis.axis_label_text_color = "black"
    
    export_png(p1, filename=filepath + indoor.iloc[0]['Location'] + '.png')
        
    tab1 = Panel(child=p1, title="Indoor Outdoor Comparison")
    
    tabs = Tabs(tabs=[ tab1])
    
    show(tabs)

    reset_output()