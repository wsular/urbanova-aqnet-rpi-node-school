#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:42:22 2022

@author: matthew
"""

from bokeh.models import Panel, Tabs
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.plotting import reset_output



def plot_indoor(indoor, site_name):

    
    p1 = figure(
            #title = location,
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m³)',
            y_range = (-5, 30))
    
    p1.title.text = site_name
    p1.title.text_font_size = '14pt'
    p1.title.text_font = 'times'
    
    p1.line(indoor.index,     indoor.PM2_5_corrected,    muted_color='black', muted_alpha=0.3,     color='black',     line_width=2) #legend='Calibrated indoor node',

    p1.legend.click_policy="mute"
    p1.legend.location='top_center'
    
    # remove bokeh and logo from final pngs (keep in so can look at data before final saved pic)
    p1.toolbar.logo = None
    p1.toolbar_location = None
    p1.xgrid.grid_line_color = None
    p1.ygrid.grid_line_color = None
    
    #p1.legend.location='top_right'
    p1.legend.label_text_font_size = "14pt"
    p1.legend.label_text_font = "times"
    p1.legend.label_text_color = "black"
    
   # p1.xaxis.axis_label="xaxis_name"
    p1.xaxis.axis_label_text_font_size = "14pt"
    p1.xaxis.major_label_text_font_size = "14pt"
    p1.xaxis.axis_label_text_font = "times"
    p1.xaxis.axis_label_text_color = "black"
    p1.xaxis.major_label_text_font = "times"

   # p1.yaxis.axis_label="yaxis_name"
    p1.yaxis.axis_label_text_font_size = "14pt"
    p1.yaxis.major_label_text_font_size = "14pt"
    p1.yaxis.axis_label_text_font = "times"
    p1.yaxis.axis_label_text_color = "black"
    p1.yaxis.major_label_text_font = "times"
    
    p1.min_border_right = 15
    
    tab1 = Panel(child=p1, title="Indoor Outdoor Comparison")
    
    tabs = Tabs(tabs=[ tab1])
    
   # show(tabs)

    reset_output()
    
    return p1