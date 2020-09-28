#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:24:27 2020

@author: matthew
"""

from bokeh.models import ColumnDataSource, Whisker
from bokeh.models import Panel, Tabs
from bokeh.plotting import figure
from bokeh.io import show

def plot_all(Audubon, Adams, Balboa, Browne, Grant, Jefferson, Lidgerwood, Regal, Sheridan, Stevens, Reference, Paccar, Augusta):
    
    p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')

    p1.title.text = 'Clarity Calibrated PM 2.5'        
    #p1.line(Audubon.index,     Audubon.PM2_5,  legend='Audubon',       color='green',     line_width=2)
    #p1.line(Adams.index,     Adams.PM2_5,  legend='Adams',       color='blue',     line_width=2)
    #p1.line(Balboa.index,     Balboa.PM2_5,  legend='Balboa',       color='red',     line_width=2)
    #1.line(Browne.index,     Browne.PM2_5,  legend='Browne',       color='black',     line_width=2)
    #p1.line(Grant.index,     Grant.PM2_5,  legend='Grant',       color='purple',     line_width=2)
    #p1.line(Jefferson.index,     Jefferson.PM2_5,  legend='Jefferson',       color='brown',     line_width=2)
    #p1.line(Lidgerwood.index,     Lidgerwood.PM2_5,  legend='Lidgerwood',       color='orange',     line_width=2)
    #p1.line(Regal.index,     Regal.PM2_5,  legend='Regal',       color='yellow',     line_width=2)
    #p1.line(Sheridan.index,     Sheridan.PM2_5,  legend='Sheridan',       color='gold',     line_width=2)
    #p1.line(Stevens.index,     Stevens.PM2_5,  legend='Stevens',       color='grey',     line_width=2)
    #p1.line(Reference.index,     Reference.PM2_5,  legend='Reference',       color='olive',     line_width=2)
    #p1.line(Paccar.index,     Paccar.PM2_5,  legend='Paccar',       color='lime',     line_width=2)
    #p1.line(Augusta.index,     Augusta.PM2_5,  legend='Augusta',       color='teal',     line_width=2)


    p1.line(Audubon.index,     Audubon.PM2_5_corrected,     legend='Audubon',        color='green',       line_width=2) #Audubon
    p1.line(Adams.index,       Adams.PM2_5_corrected,       legend='Adams',        color='blue',        line_width=2) #Adams
    p1.line(Balboa.index,      Balboa.PM2_5_corrected,      legend='Balboa',        color='red',         line_width=2) #Balboa
    p1.line(Browne.index,      Browne.PM2_5_corrected,      legend='Browne',        color='black',       line_width=2) #Browne
    p1.line(Grant.index,       Grant.PM2_5_corrected,       legend='Grant',        color='purple',      line_width=2) #Grant
    p1.line(Jefferson.index,   Jefferson.PM2_5_corrected,   legend='Jefferson',        color='brown',       line_width=2) #Jefferson
    p1.line(Lidgerwood.index,  Lidgerwood.PM2_5_corrected,  legend='Lidgerwood',        color='orange',      line_width=2) #Lidgerwood
    p1.line(Regal.index,       Regal.PM2_5_corrected,       legend='Regal',        color='khaki',       line_width=2) #Regal
    p1.line(Sheridan.index,    Sheridan.PM2_5_corrected,    legend='Sheridan',        color='deepskyblue', line_width=2) #Sheridan
    p1.line(Stevens.index,     Stevens.PM2_5_corrected,     legend='Stevens',       color='grey',        line_width=2) #Stevens
    #p1.line(Reference.index,  Reference.PM2_5_corrected,   legend='Reference',      color='olive',       line_width=2)
    #p1.line(Paccar.index,     Paccar.PM2_5_corrected,      legend='Paccar',         color='lime',        line_width=2)
    p1.line(Augusta.index,    Augusta.PM2_5,               legend='Augusta',        color='gold',        line_width=2)
   # p1.line(Audubon_Adams.index, Audubon_Adams.PM2_5_corrected, legend='Audubon1', color='gold', line_width=2)
    #p1.line(Audubon_Adams.index, Audubon_Adams.location_PM2_5_corrected, legend='Adams1', color='red', line_width=2)

    audubon_toggle = 0
    adams_toggle = 0
    balboa_toggle = 0
    browne_toggle = 0
    grant_toggle = 0
    jefferson_toggle = 0
    lidgerwood_toggle = 0
    regal_toggle = 0
    sheridan_toggle = 0
    stevens_toggle = 0
    augusta_BAM_toggle = 0

    if augusta_BAM_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Augusta.index, lower=Augusta.lower_uncertainty, upper=Augusta.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if audubon_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Audubon.index, lower=Audubon.lower_uncertainty, upper=Audubon.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if adams_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Adams.index, lower=Adams.lower_uncertainty, upper=Adams.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if balboa_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Balboa.index, lower=Balboa.lower_uncertainty, upper=Balboa.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if browne_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Browne.index, lower=Browne.lower_uncertainty, upper=Browne.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if grant_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Grant.index, lower=Grant.lower_uncertainty, upper=Grant.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if jefferson_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Jefferson.index, lower=Jefferson.lower_uncertainty, upper=Jefferson.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if lidgerwood_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Lidgerwood.index, lower=Lidgerwood.lower_uncertainty, upper=Lidgerwood.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if regal_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Regal.index, lower=Regal.lower_uncertainty, upper=Regal.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if sheridan_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Sheridan.index, lower=Sheridan.lower_uncertainty, upper=Sheridan.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass

    if stevens_toggle == 1:
        source_error = ColumnDataSource(data=dict(base=Stevens.index, lower=Stevens.lower_uncertainty, upper=Stevens.upper_uncertainty))
        p1.add_layout(
            Whisker(source=source_error, base="base", upper="upper", lower="lower")
            )
    else:
        pass
    #p1.extra_y_ranges['Snow Depth'] = Range1d(start=0, end=60)
    #p1.add_layout(LinearAxis(y_range_name='Snow Depth', axis_label='Snow Depth (in'), 'right')
    
    ###p1.line(airport.date_obj,     airport.snow_depth,     legend='Aiport Snow Depth',       color='pink',    line_width=2)
    #y_range_name = 'Snow Depth',

    p1.legend.click_policy="hide"

    #source_error = ColumnDataSource(data=dict(base=Audubon.index, lower=Audubon.lower_uncertainty, upper=Audubon.upper_uncertainty))

    #p1.add_layout(
    #    Whisker(source=source_error, base="base", upper="upper", lower="lower")
    #)

    tab1 = Panel(child=p1, title="Calibrated PM 2.5")
    
    tabs = Tabs(tabs=[ tab1])

    show(tabs)