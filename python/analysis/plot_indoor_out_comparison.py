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
from bokeh.models import Band, ColumnDataSource


def indoor_outdoor_plot(indoor, outdoor, site_number, stdev_number, time_period, shift):

        # file path based on the time period
    if time_period == '1':
        # print('1')
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_1/'
        y_scale_option = (-1.5, 30)
    elif time_period == '2':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_2/'
        y_scale_option = (-1.5, 50)
    elif time_period == '3':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_3/'
        y_scale_option = (0, 500)
    elif time_period == '4':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_4/'
        y_scale_option = (-1.5, 30)
        indoor_start_1 = '2020-09-21 20:00'
        indoor_end_1 = '2020-10-22 07:00'
        
        indoor_start_2 = '2021-01-15 07:00'
        indoor_end_2 = '2021-02-21 00:00'   # for end of analysis period
      #  indoor_end_3 = '2021-03-09 00:00'   # for end of data for sending to solmaz
      #  indoor_1 = indoor.loc[indoor_start_1:indoor_end_1]
        indoor_1 = indoor.loc[indoor_start_1:indoor_end_1]
        indoor_2 = indoor.loc[indoor_start_2:indoor_end_2]
        
        outdoor_1 = outdoor.loc[indoor_start_1:indoor_end_1]
        outdoor_2 = outdoor.loc[indoor_start_2:indoor_end_2]
        
    elif time_period == '5':
        filepath = '/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_5/'
        y_scale_option = (0, 200)
    else:
        pass

    
    # use if want to label site by school name
    #location = indoor.iloc[0]['Location']
    
    # use if want to label site by number (numbered in alphabetical order)
    
    
    p1 = figure(
            #title = location,
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)',
            y_range = y_scale_option)
    
    p1.title.text = site_number
    p1.title.text_font_size = '14pt'
    p1.title.text_font = 'times'
        
    # just plotting the calibrated shifted indoor pm2.5 so not so cluttered, unless the shift doesn't make any sense and need a visual to explain
    if shift == 'no':
        if time_period == '4':
            p1.line(indoor_1.index,     indoor_1.PM2_5_corrected,    muted_color='black', muted_alpha=0.3,     color='black',     line_width=2)#, legend='Calibrated indoor node')
            p1.line(indoor_2.index,     indoor_2.PM2_5_corrected,    muted_color='black', muted_alpha=0.3,     color='black',     line_width=2)
        else:
            p1.line(indoor.index,     indoor.PM2_5_corrected,    muted_color='black', muted_alpha=0.3,     color='black',     line_width=2) #legend='Calibrated indoor node',
    else:
        pass
    
    if time_period == '4':
        p1.line(outdoor_1.index,     outdoor_1.PM2_5_corrected,     muted_color='red', alpha=0.4,     color='red',       line_width=2)#, legend='Calibrated outdoor node')
        p1.line(outdoor_2.index,     outdoor_2.PM2_5_corrected,     muted_color='red', alpha=0.4,     color='red',       line_width=2)
    else:
        p1.line(outdoor.index,     outdoor.PM2_5_corrected,     muted_color='red', alpha=0.4,     color='red',       line_width=2) #legend='Calibrated outdoor node'
        
   # p1.line(indoor.index,     indoor.out_PM2_5_corrected,   legend='Outdoor Clarity Calibrated',  muted_color='black', alpha=0.2,     color='black',       line_width=2)
    if shift == 'yes':
        p1.line(indoor.index,     indoor.PM2_5_corrected_shift,    muted_color='black',     color='black',  line_width=2)#, alpha=0.5) legend='Calibrated indoor node',
    else:
        pass
    
    indoor['time'] = indoor.index

    source = ColumnDataSource(indoor.reset_index())

    band_indoor = Band(base= 'time', lower='lower_uncertainty', upper='upper_uncertainty', source=source, level='underlay',
            fill_alpha=0.5, line_width=1, line_color='black', fill_color = 'green')
   ### p1.add_layout(band_indoor)
    
    
    outdoor['Datetime'] = outdoor.index
    
    source2 = ColumnDataSource(outdoor.reset_index())


    band_outdoor = Band(base= 'Datetime', lower='lower_uncertainty', upper='upper_uncertainty', source=source2, level='underlay',
            fill_alpha=0.8, line_width=1, line_color='black')
   ### p1.add_layout(band_outdoor)
    
   # source_error = ColumnDataSource(data=dict(base=outdoor.index, lower=outdoor.lower_uncertainty, upper=outdoor.upper_uncertainty))

    #p1.add_layout(
    #    Whisker(source=source_error, base="base", upper="upper", lower="lower")
    #)

   # source_error_indoor = ColumnDataSource(data=dict(base=indoor.index, lower=indoor.lower_uncertainty, upper=indoor.upper_uncertainty))

    #p1.add_layout(
    #    Whisker(source=source_error_indoor, base="base", upper="upper", lower="lower")
    #)

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
    
    if shift == 'yes':
        export_png(p1, filename=filepath + indoor.iloc[0]['Location'] + '.png')
    
    elif shift == 'no':
        if stdev_number == 1:
            export_png(p1, filename=filepath + indoor.iloc[0]['Location'] + '.png')#'_unshifted_uncertainty_1''
        elif stdev_number == 2:
            export_png(p1, filename=filepath + indoor.iloc[0]['Location'] + '.png') #'_unshifted_uncertainty_2''
        
    tab1 = Panel(child=p1, title="Indoor Outdoor Comparison")
    
    tabs = Tabs(tabs=[ tab1])
    
   # show(tabs)

    reset_output()
    
    return p1