#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:51:15 2020

@author: matthew
"""

import pandas as pd
from glob import glob
from bokeh.io import export_png
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show
from bokeh.models import Panel, Tabs
from random import randint
from bokeh.models import Range1d, LinearAxis
PlotType = 'HTMLfile'

from bokeh.palettes import Category10
import itertools
from bokeh.palettes import Dark2_5 as palette

def color_gen():
    for c in itertools.cycle(Category10[14]):
        yield c
        
#%%

# Clean data and save to excel files (need different time conversion for before and after Nov 3rd for daylight savings (7 vs 8 hours))
rs = pd.DataFrame({})
#files = glob('/Users/matthew/Desktop/data/radiosondes/sept_to_Nov_3.xlsx')
#files = glob('/Users/matthew/Desktop/data/radiosondes/radiosondes_Nov_4_to_Dec_16.xlsx')
files = glob('/Users/matthew/Desktop/data/radiosondes/radiosondes_augusta_overlap.xlsx')
files.sort()
for file in files:
    rs = pd.concat([rs, pd.read_excel(file)], sort=False)


#print(rs.dtypes)
rs['tmpc'] = pd.to_numeric(rs.tmpc, errors = 'coerce')
#print(rs.dtypes)
rs = rs[rs['tmpc'].notna()]

#test = rs.iloc[1][1]
#print(test)
#print(test.tzinfo)
rs['local_time'] = rs['validUTC'].dt.tz_localize('utc').dt.tz_convert('US/Pacific') # this just displays the local time offest from UTC, the acutal value when plotted is still UTC though, so need another column with actual local time, this is useful for checking though
rs.index = rs.validUTC - pd.DateOffset(hours=8) # 8 for dates between Nov 3 2019 and Mar 8 2020 and 7 for before Nov 3
rs['true_local'] = rs.index
#rs['true_local'] = rs.local_time - pd.DateOffset(hours=8)

# save cleaned data

#rs.to_excel("/Users/matthew/Desktop/data/radiosondes/cleaned_radiosondes_Sept_1_to_Nov_3.xlsx") 
#rs.to_excel("/Users/matthew/Desktop/data/radiosondes/cleaned_radiosondes_Nov_4_to_Dec_16.xlsx") 
#rs.to_excel("/Users/matthew/Desktop/data/radiosondes/cleaned_radiosondes_Dec_17_to_March_6.xlsx") 

#use if saving as csv
#rs.to_csv('/Users/matthew/Desktop/data/radiosondes/cleaned_radiosondes_Sept_1_to_Nov_3.csv', index=False)
#rs.to_csv('/Users/matthew/Desktop/data/radiosondes/cleaned_radiosondes_Nov_4_to_Dec_16.csv', index=False)
rs.to_csv('/Users/matthew/Desktop/data/radiosondes/cleaned_radiosondes_Dec_17_to_March_6.csv', index=False)
#%%

# Choose dates of interest
    # Augusta overlap period
start_time = '2019-9-1 15:00'
end_time = '2020-03-05 23:00'
interval = '60T'
# Load Adams data for plotting

Adams_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Adams*.csv')
files.sort()
for file in files:
    Adams_All = pd.concat([Adams_All, pd.read_csv(file)], sort=False)
Adams_All['PM2_5_corrected'] = (Adams_All['PM2_5']+0.93)/1.1554
Adams_All['PM2_5_corrected'] = (Adams_All['PM2_5_corrected']+0.5693)/1.9712  # From AUGUSTA BAM comparison

Adams_All['time'] = pd.to_datetime(Adams_All['time'])
Adams_All = Adams_All.sort_values('time')
Adams_All.index = Adams_All.time
Adams = Adams_All.loc[start_time:end_time]
Adams = Adams.resample(interval).mean() 

#Read in SRCAA Augusta site BAM data for plotting

Augusta_All = pd.DataFrame({})

files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time]
#%%
# Load cleaned radiosonde data

rs = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/radiosondes/cleaned*.csv')
files.sort()
for file in files:
    rs = pd.concat([rs, pd.read_csv(file)], sort=False)

rs['true_local'] = pd.to_datetime(rs['true_local'])
rs = rs.sort_values('true_local')
rs.index = rs.true_local


rs = rs[rs.height_m < 20000]

g = rs.groupby(pd.Grouper(key='true_local', freq='W'))
weeks = [group for _,group in g]

grouped_data = []

for i,df in enumerate(weeks):
    g = df.groupby(pd.Grouper(key='true_local', freq='H'))
    days = [group for _,group in g if len(group) > 0]      # drops the empty dataframes that are created
    #days = [group for _,group in g]
    #days = [(index, group) for index, group in g if len(group) > 0]
    grouped_data.append(days)
#%%
# create a color iterator
#colors = itertools.cycle(palette) 
#color = color_gen()
colors = []
for i in range(15):
    colors.append('#%06X' % randint(0, 0xFFFFFF))
    
#%%
inv_height_list = []
inv_date_list = []
lapse_rate_list = []
df_list = []

for i,week in enumerate(grouped_data):
    
    week = grouped_data[i]
    print('one')
    print(i)
    p1 = figure(plot_width=900,
            plot_height=650,
           # x_axis_type='datetime',
            x_axis_label='Temp (ºC)',
            y_axis_label='Altitude (m)')

    p1.title.text = 'Spokane Radiosondes'
    
    for j,day in enumerate(week):
   # for day, color in zip(range(len(week)), colors): 
        
        df = week[j]
        df = df.sort_values(by=['height_m'], ascending=True)
        df.insert(0, 'row_num', range(0,len(df)))
        index = df['tmpc'].values.argmax()
        inv_temp = df.iloc[index]['tmpc']
        df['inv_height'] = df.iloc[index]['height_m']
        df['inv_height'] = df['inv_height'] - 728
        df['lapse_rate'] = -((inv_temp - df['tmpc'].iloc[0])/(df['inv_height'].iloc[0]))
        df['lapse_rate'] = df['lapse_rate']*1000 # convert to degrees C per km
        
        if df.iloc[index]['height_m'] == 728:
            del df['row_num']
            df['value'] = (df[['tmpc']] > df[['tmpc']].shift()).any(axis=1)
            df['value'] = df['value'].cumsum()
            df['value1'] = df['value'].shift(-1)
            df.insert(0, 'row_num', range(0,len(df)))
            if df['value1'].sum() > 0:
                index = (df.loc[df['value1'] == 1, 'row_num'].iloc[0])    # this gets the index of the bottom of the inversion
                df['inv_height'] = df.iloc[index]['height_m']
                df['inv_height'] = df['inv_height'] - 728
                inv_temp = (df.loc[df['value1'] == 1, 'tmpc'].iloc[0])
                df['inv_height'] = df.iloc[index]['height_m']
                df['lapse_rate'] = -((inv_temp - df['tmpc'].iloc[0])/(df['inv_height'].iloc[0]))
                df['lapse_rate'] = df['lapse_rate']*1000 # convert to degrees C per km
                inv_height_list.append(df.iloc[index]['inv_height'])
                inv_date_list.append(df.iloc[index]['true_local'])
                lapse_rate_list.append(df.iloc[index]['lapse_rate'])
               
        else:
            pass
                
        inv_height_list.append(df.iloc[index]['inv_height'])
        inv_date_list.append(df.iloc[index]['true_local'])
        lapse_rate_list.append(df.iloc[index]['lapse_rate'])
        df_list.append(df)
        #print(index)
        print('two')
        print(j)
      #  print(df)
        color = color_gen()
        p1.line(df.tmpc,     df.height_m,     legend = str(df.index[0]),    color=colors[j],  muted_color=colors[j],     line_width=2, muted_alpha=0.2) # legend = 'Day = {}'.format(day)     str(day.index[0])
   
   
    
    p1.legend.click_policy="hide"

    tab1 = Panel(child=p1, title="RadioSonde")

    tabs = Tabs(tabs=[ tab1])

    show(tabs)
#%%
inv_df = pd.DataFrame(
    {'datetime': inv_date_list, 'inv_height': inv_height_list, 'lapse_rate': lapse_rate_list})

inv_df2 = inv_df.drop_duplicates(subset=['datetime', 'inv_height'], keep='first')
inv_df2.index = inv_df2.datetime
inv_df2['Augusta_PM2_5'] = Augusta['PM2_5']
inv_df2['Adams_PM2_5'] = Adams['PM2_5']
inv_df2['Adams_temp'] = Adams['temp']
inv_df2.to_csv('/Users/matthew/Desktop/data/radiosondes/inv_height_m.csv', index=False)
#%%
    
test1 = [0]
test11 = test1[0]



p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date',
            y_axis_label='Altitude (m)')

p1.title.text = 'Test Single Day'
p1.y_range = Range1d(start=0, end=11500)
p1.line(inv_df2.index,     inv_df2.inv_height,     legend='"inversion" height',        color='green',             line_width=2, muted_color='green', muted_alpha=0.5)


p1.extra_y_ranges[' PM 2.5'] = Range1d(start=0, end=60)
p1.add_layout(LinearAxis(y_range_name=' PM 2.5', axis_label=' PM 2.5 (ug/m^3'), 'right')

p1.line(Adams.index,     Adams.PM2_5_corrected,   y_range_name = ' PM 2.5',  legend='Site #6 PM 2.5',       color='black',     muted_color='black', muted_alpha=0.3   ,    line_width=2)

p1.legend.click_policy="mute"
tab1 = Panel(child=p1, title="Radiosonde Inversion Height Time Series")



p2 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Inversion Height (m)',
            y_axis_label='PM 2.5 (ug/m3)')

p2.title.text = 'PM 2.5 vs Inversion Height'

p2.scatter(inv_df2.inv_height,     inv_df2.Augusta_PM2_5,     legend='Augusta BAM',        color='green',             line_width=2, muted_color='green', muted_alpha=0.2)
p2.scatter(inv_df2.inv_height,     inv_df2.Adams_PM2_5,     legend='#6',        color='blue',             line_width=2, muted_color='blue', muted_alpha=0.2)


p2.legend.click_policy="mute"

tab2 = Panel(child=p2, title="Inversion Height Scatter")



p3 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Lapse Rate (ºC/km)',
            y_axis_label='PM 2.5 (ug/m3)')

p3.title.text = 'PM 2.5 vs Lapse Rate'

p3.scatter(inv_df2.lapse_rate,     inv_df2.Augusta_PM2_5,    legend='Augusta',       color='green',     muted_color='green', muted_alpha=0.2   ,    line_width=2)
p3.scatter(inv_df2.lapse_rate,     inv_df2.Adams_PM2_5,    legend='#6',       color='blue',     muted_color='blue', muted_alpha=0.2   ,    line_width=2)

tab3 = Panel(child=p3, title="Lapse Rate Scatter")
p3.legend.click_policy="mute"


p4 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date',
            y_axis_label='Lapse Rate (ºC/km)')

p4.title.text = 'PM2.5 and Lapse Rate'
#p4.y_range = Range1d(start=-130, end=10000)


p4.line(Adams.index,     Adams.PM2_5_corrected,    legend='Site #6 PM 2.5',       color='black',     muted_color='black', muted_alpha=0.2   ,    line_width=2)
#p4.line(Augusta.index,     Augusta.PM2_5,    legend='Augusta PM 2.5',       color='blue',     muted_color='blue', muted_alpha=0.2   ,    line_width=2)

p4.legend.click_policy="mute"
tab4 = Panel(child=p4, title="Radiosonde Lapse Rate Time Series")

p5 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date',
            y_axis_label='Lapse Rate (ºC/km)')

p5.title.text = 'PM2.5 and Lapse Rate'

p5.line(inv_df2.index,     inv_df2.lapse_rate,     legend='lapse rate',        color='green',             line_width=2, muted_color='green', muted_alpha=0.2)
p5.legend.location='bottom_right'

p6 = gridplot([[p4], [p5]], plot_width = 700, plot_height = 300)
tab6 = Panel(child=p6, title="Radiosonde Lapse Rate Time Series")


tabs = Tabs(tabs=[ tab1, tab2, tab3, tab6])

show(tabs)



#%%

# Original Loop
for i,week in enumerate(grouped_data):
    
    week = grouped_data[i]
    print('one')
    print(i)
    p1 = figure(plot_width=900,
            plot_height=650,
           # x_axis_type='datetime',
            x_axis_label='Temp (ºC)',
            y_axis_label='Altitude (m)')

    p1.title.text = 'Spokane Radiosondes'
    
    for j,day in enumerate(week):
   # for day, color in zip(range(len(week)), colors): 
        
        df = week[j]
        df = df.sort_values(by=['height_m'], ascending=True)
        df['value'] = (df[['tmpc']] < df[['tmpc']].shift()).any(axis=1)
        df['value'] = df['value'].cumsum()
        df['value1'] = df['value'].shift(-1)
        df.insert(0, 'row_num', range(0,len(df)))
        index = (df.loc[df['value1'] == 1, 'row_num'].iloc[0])
        inv_temp = (df.loc[df['value1'] == 1, 'tmpc'].iloc[0])
        df['inv_height'] = df.iloc[index]['height_m']
        df['inv_height'] = df['inv_height'] - 728
        df['lapse_rate'] = -((inv_temp - df['tmpc'].iloc[0])/(df['inv_height'].iloc[0]-728))
        df['lapse_rate'] = df['lapse_rate']*1000 # convert to degrees C per km
        
        if df.iloc[index]['height_m'] == 728:
            del df['row_num']
            df['value'] = (df[['tmpc']] > df[['tmpc']].shift()).any(axis=1)
            df['value'] = df['value'].cumsum()
            df['value1'] = df['value'].shift(-1)
            df.insert(0, 'row_num', range(0,len(df)))
            if df['value1'].sum() > 0:
                index = (df.loc[df['value1'] == 1, 'row_num'].iloc[0])
                df['inv_height'] = df.iloc[index]['height_m']
                inv_temp = (df.loc[df['value1'] == 1, 'tmpc'].iloc[0])
                df['inv_height'] = df.iloc[index]['height_m']
                df['inv_height'] = df['inv_height'] - 728
                df['lapse_rate'] = -((inv_temp - df['tmpc'].iloc[0])/(df['inv_height'].iloc[0]-728))
                df['lapse_rate'] = df['lapse_rate']*1000 # convert to degrees C per km
                inv_height_list.append(df.iloc[index]['inv_height'])
                inv_date_list.append(df.iloc[index]['true_local'])
                lapse_rate_list.append(df.iloc[index]['lapse_rate'])
            else:
                df['inv_height'] = 10000
                inv_height_list.append(df.iloc[index]['inv_height'])
                inv_date_list.append(df.iloc[index]['true_local'])
                inv_temp = (df.loc[df['value1'] == 1, 'tmpc'].iloc[0])
                df['inv_height'] = df.iloc[index]['height_m']
                df['lapse_rate'] = -((inv_temp - df['tmpc'].iloc[0])/(df['inv_height'].iloc[0]-728))
                df['lapse_rate'] = df['lapse_rate']*1000 # convert to degrees C per km
                lapse_rate_list.append(df.iloc[index]['lapse_rate'])
                
        else:
            pass
                
        inv_height_list.append(df.iloc[index]['inv_height'])
        inv_date_list.append(df.iloc[index]['true_local'])
        lapse_rate_list.append(df.iloc[index]['lapse_rate'])
        df_list.append(df)
        #print(index)
        print('two')
        print(j)
      #  print(df)
        color = color_gen()
        p1.line(df.tmpc,     df.height_m,     legend = str(df.index[0]),    color=colors[j],  muted_color=colors[j],     line_width=2, muted_alpha=0.2) # legend = 'Day = {}'.format(day)     str(day.index[0])
   