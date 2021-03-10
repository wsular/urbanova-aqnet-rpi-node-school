#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:28:33 2021

@author: matthew
"""

import pandas as pd
from glob import glob
from figure_format import figure_format
from bokeh.models import Panel, Tabs
from bokeh.io import show
from bokeh.models import ColumnDataSource, Whisker
from bokeh.plotting import figure
from bokeh.plotting import reset_output
from bokeh.io import export_png, output_file
from bokeh.io import output_notebook, output_file, show
import numpy as np

PlotType = 'HTMLfile'
    
if PlotType=='notebook':
    output_notebook()
else:
    #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_pad_resample.html')
    output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_mean_resample.html')

#Reference node at Jefferson
start_time_1 = '2020-03-11 07:00'   
end_time_1 = '2020-10-22 07:00'

#Reference node at Audubon
start_time_2 = '2021-01-15 07:00'   
end_time_2 = '2021-02-21 07:00'

#%%

Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Ref_node_IAQU_compare/jefferson/Reference*.csv')
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)
    
Jefferson_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Ref_node_IAQU_compare/jefferson/CN*.csv')
files.sort()
for file in files:
    Jefferson_All = pd.concat([Jefferson_All, pd.read_csv(file)], sort=False)
    
Audubon_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Ref_node_IAQU_compare/audubon/CN*.csv')
files.sort()
for file in files:
    Audubon_All = pd.concat([Audubon_All, pd.read_csv(file)], sort=False)


jefferson_all = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Ref_node_IAQU_compare/jefferson/jefferson*.csv')
files.sort()
for file in files:
    jefferson_all = pd.concat([jefferson_all, pd.read_csv(file)], sort=False)
    
audubon_all = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Ref_node_IAQU_compare/audubon/audubon*.csv')
files.sort()
for file in files:
    audubon_all = pd.concat([audubon_all, pd.read_csv(file)], sort=False)
#%%

Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time

Reference_jefferson = Reference_All.loc[start_time_1:end_time_1]
Reference_audubon = Reference_All.loc[start_time_2:end_time_2]

Jefferson_All['time'] = pd.to_datetime(Jefferson_All['time'])
Jefferson_All = Jefferson_All.sort_values('time')
Jefferson_All.index = Jefferson_All.time
Jefferson = Jefferson_All.loc[start_time_1:end_time_1]

jefferson_all['Datetime'] = pd.to_datetime(jefferson_all['Datetime'])
jefferson_all = jefferson_all.sort_values('Datetime')
jefferson_all.index = jefferson_all.Datetime
jefferson = jefferson_all.loc[start_time_1:end_time_1]

Audubon_All['time'] = pd.to_datetime(Audubon_All['time'])
Audubon_All = Audubon_All.sort_values('time')
Audubon_All.index = Audubon_All.time
Audubon = Audubon_All.loc[start_time_2:end_time_2]

audubon_all['Datetime'] = pd.to_datetime(audubon_all['Datetime'])
audubon_all = audubon_all.sort_values('Datetime')
audubon_all.index = audubon_all.Datetime
audubon = audubon_all.loc[start_time_2:end_time_2]


#%%
# jefferson zoomed dates 1
#start_time_1_zoom = '2020-06-01 00:00'   
#end_time_1_zoom = '2020-09-05 00:00'

# zoomed dates 2 for smoke event (just where IAQU and ref node overlap)
#start_time_1_zoom = '2020-09-16 06:00'   
#end_time_1_zoom = '2020-09-20 00:00'

#smoke event dates for overall plot during this time
start_time_1_zoom = '2020-09-11 00:00'   
end_time_1_zoom = '2020-09-20 00:00'
#%%
# create df with all data for jefferson not in smoke
to_delete = ['2020-09-11', '2020-09-12','2020-09-13','2020-09-14','2020-09-15',
             '2020-09-16','2020-09-17','2020-09-18', '2020-09-19','2020-09-20']
Jefferson = Jefferson[~(Jefferson.index.strftime('%Y-%m-%d').isin(to_delete))]
jefferson = jefferson[~(jefferson.index.strftime('%Y-%m-%d').isin(to_delete))]
Reference_jefferson = Reference_jefferson[~(Reference_jefferson.index.strftime('%Y-%m-%d').isin(to_delete))]
#%%
Jefferson = Jefferson_All.loc[start_time_1_zoom:end_time_1_zoom]
jefferson = jefferson_all.loc[start_time_1_zoom:end_time_1_zoom]
Reference_jefferson = Reference_All.loc[start_time_1_zoom:end_time_1_zoom]

#%%
print('jefferson avg. = ', np.nanmean(jefferson['PM2_5_corrected']).round(2))
print('Jefferson avg. = ', np.nanmean(Jefferson['PM2_5_corrected']).round(2))
print('Ref Node avg. = ', np.nanmean(Reference_jefferson['PM2_5_corrected']).round(2))
print('Mean Bias = ', round(((jefferson.PM2_5_corrected - Reference_jefferson.PM2_5_corrected).sum())/len(jefferson.PM2_5_corrected),2))

print('audubon avg. = ', np.nanmean(audubon['PM2_5_corrected']).round(2))
print('Audubon avg. = ', np.nanmean(Audubon['PM2_5_corrected']).round(2))
print('Ref Node avg. = ', np.nanmean(Reference_audubon['PM2_5_corrected']).round(2))
print('Mean Bias = ', round(((audubon.PM2_5_corrected - Reference_audubon.PM2_5_corrected).sum())/len(audubon.PM2_5_corrected),2))

#median_adj.append(np.nanmedian(name['PM2_5_corrected']))
#stdev_adj.append(np.std(name['PM2_5_corrected']))
#%%

p1 = figure(
            #title = location,
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)',
        #    y_range=(0, 22) # zoom 1 y range
            )
        
# just plotting the calibrated shifted indoor pm2.5 so not so cluttered, unless the shift doesn't make any sense and need a visual to explain
p1.line(jefferson.index,     jefferson.PM2_5_corrected,  legend='Calibrated IAQU',  muted_color='black', muted_alpha=0.3,     color='black',     line_width=2)
p1.line(Reference_jefferson.index,     Reference_jefferson.PM2_5_corrected,   legend='Calibrated RN',  muted_color='red', alpha=0.4,     color='red',       line_width=2)
p1.line(Jefferson.index,     Jefferson.PM2_5_corrected,   legend='Calibrated Outdoor CN',  muted_color='black', alpha=0.2,     color='black',       line_width=2) 

#figure_format(p1)
#export_png(p1, filename='/Users/matthew/Desktop/thesis/Final_Figures/Ref_node_jefferson_compare/Ref_jefferson_zoom_1.png')
#export_png(p1, filename='/Users/matthew/Desktop/thesis/Final_Figures/Ref_node_jefferson_compare/Ref_jefferson_zoom_2.png')
#export_png(p1, filename='/Users/matthew/Desktop/thesis/Final_Figures/Ref_node_jefferson_compare/Ref_jefferson.png')
        
tab1 = Panel(child=p1, title="Indoor Jefferson Comparison")
    
tabs = Tabs(tabs=[ tab1])
    
show(tabs)

reset_output()
    
    
#%%

p1 = figure(
            #title = location,
            plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
# just plotting the calibrated shifted indoor pm2.5 so not so cluttered, unless the shift doesn't make any sense and need a visual to explain
p1.line(audubon.index,     audubon.PM2_5_corrected,  legend='Calibrated IAQU',  muted_color='black', muted_alpha=0.3,     color='black',     line_width=2)
p1.line(Reference_audubon.index,     Reference_audubon.PM2_5_corrected,   legend='Calibrated RN',  muted_color='red', alpha=0.4,     color='red',       line_width=2)
p1.line(Audubon.index,     Audubon.PM2_5_corrected,   legend='Calibrated Outdoor CN',  muted_color='black', alpha=0.2,     color='black',       line_width=2) 

figure_format(p1)
export_png(p1, filename='/Users/matthew/Desktop/thesis/Final_Figures/Ref_node_audubon_compare/Ref_audubon.png')
        
tab1 = Panel(child=p1, title="Indoor Audubon Comparison")
    
tabs = Tabs(tabs=[ tab1])
    
show(tabs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    