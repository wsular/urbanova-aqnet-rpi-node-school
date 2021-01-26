#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:43:58 2020

@author: matthew
"""


import pandas as pd
from glob import glob
from bokeh.models import Panel, Tabs
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import Range1d, LinearAxis
from Augusta_BAM_uncertainty import Augusta_BAM_uncertainty

# winter sampling time
start_time_winter = '2019-12-17 00:00'
end_time_winter = '2020-02-26 23:00'

# sept smoke sampling time
start_time_sept = '2020-09-01 00:00'
end_time_sept = '2020-09-24 23:00'

# winter 2018
start_time_2018 = '2018-10-01 00:00'
end_time_2018 = '2018-12-24 23:00'

# winter 2017
start_time_2017 = '2017-09-13 00:00'
end_time_2017 = '2017-11-15 23:00'

stdev_number = 1   # defines whether using 1 or 2 stdev for uncertainty


#%%


Monroe_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Monroe_Neph/Monroe*.csv')
files.sort()
for file in files:
    Monroe_All = pd.concat([Monroe_All, pd.read_csv(file)], sort=False)
#Monroe_All['PM2_5_corrected'] = Monroe_All['PM2_5']    # creates column with same values so loops work below in stats section

#Read in SRCAA Augusta site BAM data

Augusta_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)
Augusta_All['PM2_5_corrected'] = Augusta_All['PM2_5']    # creates column with same values so loops work below
Augusta_All['Location'] = 'Augusta'

Augusta_BAM_uncertainty(stdev_number,Augusta_All)

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta_winter = Augusta_All.loc[start_time_winter:end_time_winter] 
Augusta_sept = Augusta_All.loc[start_time_sept:end_time_sept] 
Augusta_2018 = Augusta_All.loc[start_time_2018:end_time_2018]
Augusta_2017 = Augusta_All.loc[start_time_2017:end_time_2017]


Monroe_All['time'] = pd.to_datetime(Monroe_All['time'])
Monroe_All = Monroe_All.sort_values('time')
Monroe_All.index = Monroe_All.time
Monroe_winter = Monroe_All.loc[start_time_winter:end_time_winter] 
Monroe_sept = Monroe_All.loc[start_time_sept:end_time_sept] 
Monroe_2018 = Monroe_All.loc[start_time_2018:end_time_2018] 
Monroe_2017 = Monroe_All.loc[start_time_2017:end_time_2017]

Augusta_sept_under_35 = Augusta_sept.copy()
Augusta_sept_under_35['Monroe_neph_bscat'] = Monroe_sept['bscat']

Augusta_sept_under_35 = Augusta_sept_under_35[Augusta_sept_under_35['PM2_5'] < 35]

sept_mae = Augusta_sept[['PM2_5', 'time']].copy()
sept_mae['Monroe'] = Monroe_sept['PM2_5']
sept_mae = sept_mae[sept_mae['PM2_5'] < 35]

monroe_2019_winter_mae = ((Augusta_winter.PM2_5_corrected - Monroe_winter.PM2_5).sum())/len(Monroe_winter.index)
print('Neph 2019 winter MAE = ', monroe_2019_winter_mae)

monroe_2018_winter_mae = ((Augusta_2018.PM2_5_corrected - Monroe_2018.PM2_5).sum())/len(Monroe_2018.index)
print('Neph 2018 winter MAE = ', monroe_2018_winter_mae)

monroe_2017_winter_mae = ((Augusta_2017.PM2_5_corrected - Monroe_2017.PM2_5).sum())/len(Monroe_2017.index)
print('Neph 2017 winter MAE = ', monroe_2017_winter_mae)

monroe_2020_sept_mae = ((sept_mae.PM2_5 - sept_mae.Monroe).sum())/len(sept_mae.index)
print('Neph 2020 Sept MAE = ', monroe_2020_sept_mae)

#%%



p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='bscat (m^-1)')

p1.title.text = 'Nephelometer Back Scatter Coefficients'    

p1.extra_y_ranges['bscat'] = Range1d(start=-1, end=25)
p1.add_layout(LinearAxis(y_range_name='bscat', axis_label=' PM2.5 (ug/m^3)'), 'right')


#p1.line(Monroe_winter.index,     Monroe_winter.bscat,  legend='Winter bscat',     color='red',   muted_color='red', muted_alpha=0.1 , line_width=2)
p1.line(Monroe_winter.index,     Monroe_winter.PM2_5,  legend='Winter PM 2.5',       color='black',   muted_color='black', muted_alpha=0.1,  line_width=2)
p1.line(Monroe_winter.index,     Monroe_winter.bscat,  legend='bscat',   y_range_name = 'bscat',      color='red',  muted_color='red', muted_alpha=0.1 ,  line_width=2)
p1.line(Augusta_winter.index,     Augusta_winter.PM2_5,  legend='BAM PM 2.5',         color='green',  muted_color='green', muted_alpha=0.1 ,  line_width=2)



#p1.line(Monroe_sept.index,     Monroe_sept.bscat,  legend='September bscat',     color='red',   muted_color='red', muted_alpha=0.1 , line_width=2)
p1.line(Monroe_sept.index,     Monroe_sept.PM2_5,  legend='September PM 2.5',       color='black',   muted_color='black', muted_alpha=0.2,  line_width=2)
p1.line(Monroe_sept.index,     Monroe_sept.bscat,  legend='bscat',   y_range_name = 'bscat',      color='red',  muted_color='red', muted_alpha=0.1 ,  line_width=2)
p1.line(Augusta_sept.index,     Augusta_sept.PM2_5,  legend='BAM PM 2.5',        color='green',  muted_color='green', muted_alpha=0.1 ,  line_width=2)

p1.line(Monroe_All.index,     Monroe_All.PM2_5,  legend='All Neph PM 2.5',       color='blue',   muted_color='blue', muted_alpha=0.2,  line_width=2)
p1.line(Monroe_All.index,     Monroe_All.bscat,  legend='All bscat',   y_range_name = 'bscat',      color='brown',  muted_color='brown', muted_alpha=0.1 ,  line_width=2)
p1.line(Augusta_All.index,     Augusta_All.PM2_5,  legend='All BAM PM 2.5',        color='gray',  muted_color='gray', muted_alpha=0.1 ,  line_width=2)



p1.legend.click_policy="mute"


tab1 = Panel(child=p1, title="bscat and Neph PM2.5 time series")





p2 = figure(plot_width=900,
            plot_height=450,
       #     x_axis_type='datetime',
            x_axis_label='bscat (m^-1',
            y_axis_label='PM 2.5 (ug/m^3)')

p2.title.text = 'bscat vs Neph PM2.5'    


p2.scatter(Monroe_winter.bscat,     Monroe_winter.PM2_5,  legend='2019 winter',     color='red',   muted_color='red', muted_alpha=0.2 , line_width=2)
p2.scatter(Monroe_sept.bscat,     Monroe_sept.PM2_5,  legend='september',     color='black',   muted_color='black', muted_alpha=0.2 , line_width=2)
#p2.scatter(Monroe_All.bscat,     Monroe_All.PM2_5,  legend='All',     color='gray',   muted_color='gray', muted_alpha=0.1 , line_width=2)
p2.scatter(Monroe_2018.bscat,     Monroe_2018.PM2_5,  legend='2018 winter',     color='green',   muted_color='green', muted_alpha=0.2 , line_width=2)
p2.scatter(Monroe_2017.bscat,     Monroe_2017.PM2_5,  legend='2017 winter',     color='orange',   muted_color='orange', muted_alpha=0.2 , line_width=2)


p2.legend.click_policy="mute"
tab2 = Panel(child=p2, title="bscat vs PM 2.5")



p3 = figure(plot_width=900,
            plot_height=450,
       #     x_axis_type='datetime',
            x_axis_label='bscat (m^-1',
            y_axis_label='PM 2.5 (ug/m^3)')

p3.title.text = 'bscat vs BAM PM2.5'    


p3.scatter(Monroe_winter.bscat,     Augusta_winter.PM2_5,  legend='winter',     color='red',   muted_color='red', muted_alpha=0.1 , line_width=2)
p3.scatter(Monroe_sept.bscat,     Augusta_sept.PM2_5,  legend='september',     color='black',   muted_color='black', muted_alpha=0.1 , line_width=2)
#p3.scatter(Monroe_2018.bscat,     Augusta_2018.PM2_5,  legend='winter 2018',     color='green',   muted_color='gree', muted_alpha=0.1 , line_width=2)



p3.legend.click_policy="mute"
tab3 = Panel(child=p3, title="bscat vs BAM PM 2.5")




p4 = figure(plot_width=900,
            plot_height=450,
       #     x_axis_type='datetime',
            x_axis_label='BAM PM 2.5 (ug/m^3)',
            y_axis_label='bscat (m^-1)')

p4.title.text = 'bscat vs BAM PM2.5'    


p4.scatter(Augusta_winter.PM2_5,       Monroe_winter.bscat,       legend='winter',     color='red',   muted_color='red', muted_alpha=0.1 , line_width=2)
p4.scatter(Augusta_sept_under_35.PM2_5,       Augusta_sept_under_35.Monroe_neph_bscat,       legend='september',     color='black',   muted_color='black', muted_alpha=0.1 , line_width=2)



p4.legend.click_policy="hide"
tab4 = Panel(child=p4, title="bscat vs BAM PM 2.5 under 35ug/m^3")


p5 = figure(plot_width=900,
            plot_height=450,
       #     x_axis_type='datetime',
            x_axis_label='BAM PM 2.5 (ug/m^3)',
            y_axis_label='Monroe Neph PM 2.5 (ug/m^3)')

p5.title.text = 'Neph vs BAM PM2.5'    


p5.scatter(Augusta_winter.PM2_5_corrected,   Monroe_winter.PM2_5,       legend='winter 2020',     color='red',   muted_color='red', muted_alpha=0.2 , line_width=2)
p5.scatter(Augusta_sept.PM2_5_corrected,     Monroe_sept.PM2_5,       legend='september 2020',     color='black',   muted_color='black', muted_alpha=0.2 , line_width=2)
p5.scatter(Augusta_2018.PM2_5_corrected,     Monroe_2018.PM2_5,       legend='winter 2018',     color='green',   muted_color='green', muted_alpha=0.2 , line_width=2)
p5.scatter(Augusta_2017.PM2_5_corrected,     Monroe_2017.PM2_5,       legend='winter 2017',     color='gray',   muted_color='gray', muted_alpha=0.2 , line_width=2)



p5.legend.click_policy="mute"
tab5 = Panel(child=p5, title="Neph vs BAM PM 2.5")


tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5])

show(tabs)