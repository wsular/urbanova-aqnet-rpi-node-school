#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:53:33 2019

@author: matthew
"""
import requests
import pandas as pd
import datetime
import json
from glob import glob
import numpy as np


# Load in EBAM Data
EBAM_data = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/calibration/Reference/EBAM*.csv')
files.sort()
for file in files:
    EBAM_data = pd.concat([EBAM_data, pd.read_csv(file)], sort=False)


EBAM_data.rename(columns={'ConcHR(ug/m3)':'PM2_5'}, inplace=True)
EBAM_data.rename(columns={'BP(mmHg)':'P'}, inplace=True)
EBAM_data.rename(columns={'RH(%)':'RH'}, inplace=True)
EBAM_data.rename(columns={'AT(C)':'Temp'}, inplace=True)

#convert EBAM Pressure from mmHg to mb 
EBAM_data.loc[:,'P'] *= 1.33322

#convert times to objects for plotting
EBAM_data['Time'] = pd.to_datetime(EBAM_data['Time'])
EBAM_data.index = EBAM_data.Time
#print (EBAM_data.dtypes)
#%%
#Create data frame for just the indoor unit and EBAM overlap
EBAM_indoor_overlap= EBAM_data.loc['2019-7-25 12:00':'2019-7-31']
#%%
#Create data frame for just the Clarity batch 1 nodes and EBAM overlap
EBAM_Clarity_batch_1_overlap = EBAM_data.loc['2019-7-12 16:00':'2019-7-31 14:00']     #change to exact hours needed for each batch 1 sensor for direct hourly comparison
del EBAM_Clarity_batch_1_overlap['Time']
#EBAM_Clarity_batch_1_overlap.reset_index( inplace = True)
#%%
#Create data frame for just the Clarity batch 2 nodes and EBAM overlap
EBAM_Clarity_batch_2_overlap = EBAM_data.loc['2019-8-19':'2019-8-22']
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#Load 1st batch Paccar roof overlap data, change times to objects for plotting, order based on time so axis dates match with EBAM (otherwise all zigzagged back and forth in plots)

with open('/Users/matthew/Desktop/data/calibration/Clarity/Audubon/Audubon_paccar_roof.json') as json_file:
    Audubon_Clarity=json.load(json_file)
Audubon_Clarity = pd.DataFrame(Audubon_Clarity)
Audubon_Clarity['time'] = pd.to_datetime(Audubon_Clarity['time'])
Audubon_Clarity = Audubon_Clarity.sort_values('time')                         # reorder to descending order so go from distant to recent date in .loc selection (otherwise will be blank and need to input it backwards)
Audubon_Clarity.index = Audubon_Clarity.time
Audubon_Clarity = Audubon_Clarity.loc['2019-7-12 16:00':'2019-7-31']          # match up with EBAM (start two hours after background set)
Hourly_Audubon = Audubon_Clarity.resample('H').mean()                         #creates new data frame averaging data every hour
Hourly_Audubon.rename(columns={'PM2_5':'PM2_5_Audubon'}, inplace=True)
Hourly_Audubon.reset_index( inplace=True)                                     #reset index so can combine with EBAM Dataframe

#%%
#Combine the two dataframes into one
Hourly_Audubon1 = pd.concat([Hourly_Audubon, EBAM_Clarity_batch_1_overlap], axis=1)

#%%
#del Hourly_Audubon1['Time']
#%%
Hourly_Audubon1[Hourly_Audubon1.index.duplicated()]    # look for duplicate values
#%%
Hourly_Audubon1 = Hourly_Audubon1[Hourly_Audubon1['PM2_5'] < 300]   #remove extremely high EBAM values
#%%
#indexNames = Hourly_Audubon[Hourly_Audubon['PM10'] > 500 ].index

# Delete these row indexes from dataFrame
#Hourly_Audubon.drop(indexNames , inplace=True)
#print (Audubon_Clarity.dtypes)

with open('/Users/matthew/Desktop/data/calibration/Clarity/Balboa/Balboa_paccar_roof.json') as json_file:
    Balboa_Clarity=json.load(json_file)
Balboa_Clarity = pd.DataFrame(Balboa_Clarity)
Balboa_Clarity['time'] = pd.to_datetime(Balboa_Clarity['time'])
Balboa_Clarity = Balboa_Clarity.sort_values('time')
Balboa_Clarity.index = Balboa_Clarity.time

with open('/Users/matthew/Desktop/data/calibration/Clarity/Browne/Browne_paccar_roof.json') as json_file:
    Browne_Clarity=json.load(json_file)
Browne_Clarity = pd.DataFrame(Browne_Clarity)
Browne_Clarity['time'] = pd.to_datetime(Browne_Clarity['time'])
Browne_Clarity = Browne_Clarity.sort_values('time')
Browne_Clarity.index = Browne_Clarity.time

with open('/Users/matthew/Desktop/data/calibration/Clarity/Lidgerwood/Lidgerwood_paccar_roof.json') as json_file:
    Lidgerwood_Clarity=json.load(json_file)
Lidgerwood_Clarity = pd.DataFrame(Lidgerwood_Clarity)
Lidgerwood_Clarity['time'] = pd.to_datetime(Lidgerwood_Clarity['time'])
Lidgerwood_Clarity = Lidgerwood_Clarity.sort_values('time')
Lidgerwood_Clarity.index = Lidgerwood_Clarity.time
Lidgerwood_Clarity= Lidgerwood_Clarity.loc['2019-07-12':'2019-07-31']        # remove the random June data for Lidgerwood

with open('/Users/matthew/Desktop/data/calibration/Clarity/Regal/Regal_paccar_roof.json') as json_file:
    Regal_Clarity=json.load(json_file)
Regal_Clarity = pd.DataFrame(Regal_Clarity)
Regal_Clarity['time'] = pd.to_datetime(Regal_Clarity['time'])
Regal_Clarity = Regal_Clarity.sort_values('time')
Regal_Clarity.index = Regal_Clarity.time
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#Load 2nd batch Paccar roof overlap data, change times to objects for plotting, order based on time so axis dates match with EBAM (otherwise all zigzagged back and forth in plots)

with open('/Users/matthew/Desktop/data/calibration/Clarity/Adams/Adams_paccar_roof.json') as json_file:
    Adams_Clarity=json.load(json_file)
Adams_Clarity = pd.DataFrame(Adams_Clarity)
Adams_Clarity['time'] = pd.to_datetime(Adams_Clarity['time'])
Adams_Clarity = Adams_Clarity.sort_values('time')
Adams_Clarity.index = Adams_Clarity.time
Hourly_Adams = Adams_Clarity.resample('H').mean()                         #creates new data frame averaging data every hour
#Hourly_Adams = Adams_Clarity.groupby(Adams_Clarity.index.hour).mean()    #creates new data frame with average for each hour of the day (24 hours total)

with open('/Users/matthew/Desktop/data/calibration/Clarity/Grant/Grant_paccar_roof.json') as json_file:
    Grant_Clarity=json.load(json_file)
Grant_Clarity = pd.DataFrame(Grant_Clarity)
Grant_Clarity['time'] = pd.to_datetime(Grant_Clarity['time'])
Grant_Clarity = Grant_Clarity.sort_values('time')
Grant_Clarity.index = Grant_Clarity.time

with open('/Users/matthew/Desktop/data/calibration/Clarity/Jefferson/Jefferson_paccar_roof.json') as json_file:
    Jefferson_Clarity=json.load(json_file)
Jefferson_Clarity = pd.DataFrame(Jefferson_Clarity)
Jefferson_Clarity['time'] = pd.to_datetime(Jefferson_Clarity['time'])
Jefferson_Clarity = Jefferson_Clarity.sort_values('time')
Jefferson_Clarity.index = Jefferson_Clarity.time

with open('/Users/matthew/Desktop/data/calibration/Clarity/Sheridan/Sheridan_paccar_roof.json') as json_file:
    Sheridan_Clarity=json.load(json_file)
Sheridan_Clarity = pd.DataFrame(Sheridan_Clarity)
Sheridan_Clarity['time'] = pd.to_datetime(Sheridan_Clarity['time'])
Sheridan_Clarity = Sheridan_Clarity.sort_values('time')
Sheridan_Clarity.index = Sheridan_Clarity.time
Sheridan_Clarity= Sheridan_Clarity.loc['2019-07-12':'2019-07-31']        # remove the random June data for Lidgerwood

with open('/Users/matthew/Desktop/data/calibration/Clarity/Stevens/Stevens_paccar_roof.json') as json_file:
    Stevens_Clarity=json.load(json_file)
Stevens_Clarity = pd.DataFrame(Stevens_Clarity)
Stevens_Clarity['time'] = pd.to_datetime(Stevens_Clarity['time'])
Stevens_Clarity = Stevens_Clarity.sort_values('time')
Stevens_Clarity.index = Stevens_Clarity.time
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#Load indoor unit calibration overlap and convert dates to objects for plotting

with open('/Users/matthew/Desktop/data/calibration/Adams/WSU_LAR_Indoor_Air_Quality_Node_9_20190731_000113.json') as json_file:
    Adams_Calibration=json.load(json_file)
adams = pd.DataFrame.from_dict(Adams_Calibration)
adams['Datetime'] = pd.to_datetime(adams['Datetime'])
adams.index = adams.Datetime

with open('/Users/matthew/Desktop/data/calibration/Audubon/WSU_LAR_Indoor_Air_Quality_Node_11_20190731_000033.json') as json_file:
    Audubon_Calibration = json.load(json_file)
audubon = pd.DataFrame.from_dict(Audubon_Calibration)
audubon['Datetime'] = pd.to_datetime(audubon['Datetime'])
audubon.index = audubon.Datetime

with open('/Users/matthew/Desktop/data/calibration/Balboa/WSU_LAR_Indoor_Air_Quality_Node_6_20190731_000134.json') as json_file:
    Balboa_Calibration = json.load(json_file)
balboa = pd.DataFrame.from_dict(Balboa_Calibration)
balboa['Datetime'] = pd.to_datetime(balboa['Datetime'])
balboa.index = balboa.Datetime

with open('/Users/matthew/Desktop/data/calibration/Browne/WSU_LAR_Indoor_Air_Quality_Node_5_20190731_000000.json') as json_file:
    Browne_Calibration = json.load(json_file)
browne = pd.DataFrame.from_dict(Browne_Calibration)
browne['Datetime'] = pd.to_datetime(browne['Datetime'])
browne.index = browne.Datetime

with open('/Users/matthew/Desktop/data/calibration/Grant/WSU_LAR_Indoor_Air_Quality_Node_4_20190731_000102.json') as json_file:
    Grant_Calibration = json.load(json_file)
grant = pd.DataFrame.from_dict(Grant_Calibration)
grant['Datetime'] = pd.to_datetime(grant['Datetime'])
grant.index = grant.Datetime

with open('/Users/matthew/Desktop/data/calibration/Jefferson/WSU_LAR_Indoor_Air_Quality_Node_8_20190731_000000.json') as json_file:
    Jefferson_Calibration=json.load(json_file)
jefferson = pd.DataFrame.from_dict(Jefferson_Calibration)
jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson.index = jefferson.Datetime

with open('/Users/matthew/Desktop/data/calibration/Lidgerwood/WSU_LAR_Indoor_Air_Quality_Node_3_20190731_000156.json') as json_file:
    Lidgerwood_Calibration = json.load(json_file)
lidgerwood = pd.DataFrame.from_dict(Lidgerwood_Calibration)
lidgerwood['Datetime'] = pd.to_datetime(lidgerwood['Datetime'])
lidgerwood.index = lidgerwood.Datetime

with open('/Users/matthew/Desktop/data/calibration/Regal/WSU_LAR_Indoor_Air_Quality_Node_10_20190731_000016.json') as json_file:
    Regal_Calibration = json.load(json_file)
regal = pd.DataFrame.from_dict(Regal_Calibration)
regal['Datetime'] = pd.to_datetime(regal['Datetime'])
regal.index = regal.Datetime

with open('/Users/matthew/Desktop/data/calibration/Sheridan/WSU_LAR_Indoor_Air_Quality_Node_1_20190731_000012.json') as json_file:
    Sheridan_Calibration = json.load(json_file)
sheridan = pd.DataFrame.from_dict(Sheridan_Calibration)
sheridan['Datetime'] = pd.to_datetime(sheridan['Datetime'])
sheridan.index = sheridan.Datetime   
 
with open('/Users/matthew/Desktop/data/calibration/Stevens/WSU_LAR_Indoor_Air_Quality_Node_2_20190731_000023.json') as json_file:
    Stevens_Calibration = json.load(json_file)
stevens = pd.DataFrame.from_dict(Stevens_Calibration)
stevens['Datetime'] = pd.to_datetime(stevens['Datetime'])
stevens.index = stevens.Datetime
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure

PlotType = 'HTMLfile'

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/Clarity_nodes_vs_EBAM_PM2_5.html')


#the data
x=np.array(Hourly_Audubon1.PM2_5)
y=np.array(Hourly_Audubon1.PM2_5_Audubon)

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

# plot it
p1 = figure(plot_width=900,
            plot_height=450,
            #x_axis_type='EBAM (ug/m^3',
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Audubon_Clarity (ug/m^3)')
            
p1.circle(x,y)
p1.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
#p1.scatter(EBAM_Clarity_batch_1_overlap.PM10,  Hourly_Audubon.PM10,                legend='PM10',       color='gold',     line_width=2)
p1.legend.location='top_left'


tab1 = Panel(child=p1, title="Audubon Clarity PM2_5")

tabs = Tabs(tabs=[ tab1])

show(tabs)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/Clarity_batch_1_overlap.html')
#Plot Clarity Batch 1 and EBAM overlap data

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='Temperature (C)')
            
p1.line(EBAM_Clarity_batch_1_overlap.index,  EBAM_Clarity_batch_1_overlap.Temp,  legend='EBAM',       color='gold',     line_width=2)
p1.line(Audubon_Clarity.index,               Audubon_Clarity.temp,               legend='Audubon',    color='red',      line_width=2)
p1.line(Balboa_Clarity.index,                Balboa_Clarity.temp,                legend='Balboa',     color='green',    line_width=2)
p1.line(Browne_Clarity.index,                Browne_Clarity.temp,                legend='Browne',     color='black',     line_width=2)
p1.line(Lidgerwood_Clarity.index,            Lidgerwood_Clarity.temp,            legend='Lidgerwood', color='cyan',     line_width=2)
p1.line(Regal_Clarity.index,                 Regal_Clarity.temp,                 legend='Regal',      color='gray',  line_width=2)
p1.legend.location='top_left'
tab1 = Panel(child=p1, title="Temperature")

p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')
            
p2.line(EBAM_Clarity_batch_1_overlap.index,  EBAM_Clarity_batch_1_overlap.RH,  legend='EBAM',       color='gold',     line_width=2)
p2.line(Audubon_Clarity.index,               Audubon_Clarity.Rel_humid,        legend='Audubon',    color='red',      line_width=2)
p2.line(Balboa_Clarity.index,                Balboa_Clarity.Rel_humid,         legend='Balboa',     color='green',    line_width=2)
p2.line(Browne_Clarity.index,                Browne_Clarity.Rel_humid,         legend='Browne',     color='black',     line_width=2)
p2.line(Lidgerwood_Clarity.index,            Lidgerwood_Clarity.Rel_humid,     legend='Lidgerwood', color='cyan',     line_width=2)
p2.line(Regal_Clarity.index,                 Regal_Clarity.Rel_humid,          legend='Regal',      color='gray',  line_width=2)
p2.legend.location='top_left'
tab2 = Panel(child=p2, title="RH")

p3 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m^3)')
            
p3.line(EBAM_Clarity_batch_1_overlap.index,    EBAM_Clarity_batch_1_overlap.PM2_5,  legend='EBAM',       color='gold',     line_width=2)
p3.line(Audubon_Clarity.index,                 Audubon_Clarity.PM2_5,               legend='Audubon',    color='red',      line_width=2)
p3.line(Balboa_Clarity.index,                  Balboa_Clarity.PM2_5,                legend='Balboa',     color='green',    line_width=2)
p3.line(Browne_Clarity.index,                  Browne_Clarity.PM2_5,                legend='Browne',     color='black',     line_width=2)
p3.line(Lidgerwood_Clarity.index,              Lidgerwood_Clarity.PM2_5,            legend='Lidgerwood', color='cyan',     line_width=2)
p3.line(Regal_Clarity.index,                   Regal_Clarity.PM2_5,                 legend='Regal',      color='gray',  line_width=2)
p3.legend.location='top_left'
tab3 = Panel(child=p3, title="PM2_5")

tabs = Tabs(tabs=[ tab1, tab2, tab3 ])

show(tabs)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/Clarity_batch_2_overlap.html')
#Plot Clarity Batch 2 and EBAM overlap data

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='Temperature (C)')
            
p1.line(EBAM_Clarity_batch_2_overlap.index,    EBAM_Clarity_batch_2_overlap.Temp,    legend='EBAM',       color='gold',     line_width=2)
p1.line(Adams_Clarity.index,                   Adams_Clarity.temp,                   legend='Adams',      color='red',      line_width=2)
p1.line(Grant_Clarity.index,                   Grant_Clarity.temp,                   legend='Grant',      color='green',    line_width=2)
p1.line(Jefferson_Clarity.index,               Jefferson_Clarity.temp,               legend='Jefferson',  color='black',    line_width=2)
p1.line(Sheridan_Clarity.index,                Sheridan_Clarity.temp,                legend='Sheridan',   color='cyan',     line_width=2)
p1.line(Stevens_Clarity.index,                 Stevens_Clarity.temp,                 legend='Stevens',    color='gray',     line_width=2)
p1.legend.location='top_left'
tab1 = Panel(child=p1, title="Temperature")

p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')
            
p2.line(EBAM_Clarity_batch_2_overlap.index,   EBAM_Clarity_batch_2_overlap.RH,   legend='EBAM',       color='gold',     line_width=2)
p2.line(Adams_Clarity.index,                  Adams_Clarity.Rel_humid,           legend='Adams',      color='red',      line_width=2)
p2.line(Grant_Clarity.index,                  Grant_Clarity.Rel_humid,           legend='Grant',      color='green',    line_width=2)
p2.line(Jefferson_Clarity.index,              Jefferson_Clarity.Rel_humid,       legend='Jefferson',  color='black',    line_width=2)
p2.line(Sheridan_Clarity.index,               Sheridan_Clarity.Rel_humid,        legend='Sheridan',   color='cyan',     line_width=2)
p2.line(Stevens_Clarity.index,                Stevens_Clarity.Rel_humid,         legend='Stevens',    color='gray',     line_width=2)
p2.legend.location='top_left'
tab2 = Panel(child=p2, title="RH")

p3 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM10 (ug/m^3)')
            
p3.line(EBAM_Clarity_batch_2_overlap.index,    EBAM_Clarity_batch_2_overlap.PM2_5,  legend='EBAM',       color='gold',     line_width=2)
p3.line(Adams_Clarity.index,                   Adams_Clarity.PM2_5,                 legend='Adams',      color='red',      line_width=2)
p3.line(Grant_Clarity.index,                   Grant_Clarity.PM2_5,                 legend='Grant',      color='green',    line_width=2)
p3.line(Jefferson_Clarity.index,               Jefferson_Clarity.PM2_5,             legend='Jefferson',  color='black',    line_width=2)
p3.line(Sheridan_Clarity.index,                Sheridan_Clarity.PM2_5,              legend='Sheridan',   color='cyan',     line_width=2)
p3.line(Stevens_Clarity.index,                 Stevens_Clarity.PM2_5,               legend='Stevens',    color='gray',     line_width=2)
p3.legend.location='top_left'
tab3 = Panel(child=p3, title="PM2_5")

tabs = Tabs(tabs=[ tab1, tab2, tab3 ])

show(tabs)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#compare Clarity Batch 1 and Indoor Sensors
if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/sensors_overlap.html')

#........Temp
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='Temperature (C)')
            
p1.line(adams.index,                         adams.Temp,                         legend='Adams',              color='blue',     line_width=2)
p1.line(audubon.index,                       audubon.Temp,                       legend='Audubon',            color='red',      line_width=2)
p1.line(balboa.index,                        balboa.Temp,                        legend='Balboa',             color='green',    line_width=2)
p1.line(browne.index,                        browne.Temp,                        legend='Browne',             color='gray',     line_width=2)
p1.line(grant.index,                         grant.Temp,                         legend='Grant',              color='cyan',     line_width=2)
p1.line(jefferson.index,                     jefferson.Temp,                     legend='Jefferson',          color='magenta',  line_width=2)
p1.line(lidgerwood.index,                    lidgerwood.Temp,                    legend='Lidgerwood',         color='black',    line_width=2)
p1.line(regal.index,                         regal.Temp,                         legend='Regal',              color='orange',   line_width=2)
p1.line(sheridan.index,                      sheridan.Temp,                      legend='Sheridan',           color='lightgreen', line_width=2)
p1.line(stevens.index,                       stevens.Temp,                       legend='Stevens',            color='olive',    line_width=2)
p1.line(Audubon_Clarity.index,               Audubon_Clarity.temp,               legend='Audubon_Outdoor',    color='peru',      line_width=2)
p1.line(Balboa_Clarity.index,                Balboa_Clarity.temp,                legend='Balboa_Outdoor',     color='navy',    line_width=2)
p1.line(Browne_Clarity.index,                Browne_Clarity.temp,                legend='Browne_Outdoor',     color='teal',     line_width=2)
p1.line(Lidgerwood_Clarity.index,            Lidgerwood_Clarity.temp,            legend='Lidgerwood_Outdoor', color='seashell',     line_width=2)
p1.line(Regal_Clarity.index,                 Regal_Clarity.temp,                 legend='Regal_Outdoor',      color='violet',  line_width=2)
p1.line(EBAM_indoor_overlap.index,           EBAM_indoor_overlap.Temp,           legend='EBAM',               color='gold',     line_width=2)
p1.legend.location='top_left'

tab1 = Panel(child=p1, title="Temperature")


# ....Relative Humidity
p2 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')
p2.line(adams.index,                         adams.RH,                         legend='Adams',      color='blue',     line_width=2)
p2.line(audubon.index,                       audubon.RH,                       legend='Audubon',    color='red',      line_width=2)
p2.line(balboa.index,                        balboa.RH,                        legend='Balboa',     color='green',    line_width=2)
p2.line(browne.index,                        browne.RH,                        legend='Browne',     color='gray',     line_width=2)
p2.line(grant.index,                         grant.RH,                         legend='Grant',      color='cyan',     line_width=2)
p2.line(jefferson.index,                     jefferson.RH,                     legend='Jefferson',  color='magenta',  line_width=2)
p2.line(lidgerwood.index,                    lidgerwood.RH,                    legend='Lidgerwood', color='black',    line_width=2)
p2.line(regal.index,                         regal.RH,                         legend='Regal',      color='orange',   line_width=2)
p2.line(sheridan.index,                      sheridan.RH,                      legend='Sheridan',   color='lightgreen', line_width=2)
p2.line(stevens.index,                       stevens.RH,                       legend='Stevens',    color='olive',    line_width=2)
p2.line(Audubon_Clarity.index,               Audubon_Clarity.Rel_humid,        legend='Audubon_Outdoor',    color='peru',      line_width=2)
p2.line(Balboa_Clarity.index,                Balboa_Clarity.Rel_humid,         legend='Balboa_Outdoor',     color='navy',    line_width=2)
p2.line(Browne_Clarity.index,                Browne_Clarity.Rel_humid,         legend='Browne_Outdoor',     color='teal',     line_width=2)
p2.line(Lidgerwood_Clarity.index,            Lidgerwood_Clarity.Rel_humid,     legend='Lidgerwood_Outdoor', color='seashell',     line_width=2)
p2.line(Regal_Clarity.index,                 Regal_Clarity.Rel_humid,          legend='Regal_Outdoor',      color='violet',  line_width=2)
p2.line(EBAM_indoor_overlap.index,           EBAM_indoor_overlap.RH,           legend='EBAM',       color='gold',     line_width=2)
p2.legend.location='top_left'
tab2 = Panel(child=p2, title="Relative Humidity")

#.........PM10
p3 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 10 (ug m-3)')
p3.line(adams.index,                           adams.PM10_standard,                legend='Adams',      color='blue',     line_width=2)
p3.line(audubon.index,                         audubon.PM10_standard,              legend='Audubon',    color='red',      line_width=2)
p3.line(balboa.index,                          balboa.PM10_standard,               legend='Balboa',     color='green',    line_width=2)
p3.line(browne.index,                          browne.PM10_standard,               legend='Browne',     color='gray',     line_width=2)
p3.line(grant.index,                           grant.PM10_standard,                legend='Grant',      color='cyan',     line_width=2)
p3.line(jefferson.index,                       jefferson.PM10_standard,            legend='Jefferson',  color='magenta',  line_width=2)
p3.line(lidgerwood.index,                      lidgerwood.PM10_standard,           legend='Lidgerwood', color='black',    line_width=2)
p3.line(regal.index,                           regal.PM10_standard,                legend='Regal',      color='orange',   line_width=2)
p3.line(sheridan.index,                        sheridan.PM10_standard,             legend='Sheridan',   color='lightgreen', line_width=2)
p3.line(stevens.index,                         stevens.PM10_standard,              legend='Stevens',    color='olive',    line_width=2)
p3.line(Audubon_Clarity.index,                 Audubon_Clarity.PM10,               legend='Audubon_Outdoor',    color='peru',      line_width=2)
p3.line(Balboa_Clarity.index,                  Balboa_Clarity.PM10,                legend='Balboa_Outdoor',     color='navy',    line_width=2)
p3.line(Browne_Clarity.index,                  Browne_Clarity.PM10,                legend='Browne_Outdoor',     color='teal',     line_width=2)
p3.line(Lidgerwood_Clarity.index,              Lidgerwood_Clarity.PM10,            legend='Lidgerwood_Outdoor', color='seashell',     line_width=2)
p3.line(Regal_Clarity.index,                   Regal_Clarity.PM10,                 legend='Regal_Outdoor',      color='violet',      line_width=2)
p3.legend.location='top_left'
tab3 = Panel(child=p3, title="PM 10")

tabs = Tabs(tabs=[ tab1, tab2, tab3])

show(tabs)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/Indoor_units_overlap.html')
#Indoor Overlap Data
# ....Temperature
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='Temperature (C)')
            
p1.line(adams.index,      adams.Temp,      legend='Adams',      color='blue',     line_width=2)
p1.line(audubon.index,    audubon.Temp,    legend='Audubon',    color='red',      line_width=2)
p1.line(balboa.index,     balboa.Temp,     legend='Balboa',     color='green',    line_width=2)
p1.line(browne.index,     browne.Temp,     legend='Browne',     color='gray',     line_width=2)
p1.line(grant.index,      grant.Temp,      legend='Grant',      color='cyan',     line_width=2)
p1.line(jefferson.index,  jefferson.Temp,  legend='Jefferson',  color='magenta',  line_width=2)
p1.line(lidgerwood.index, lidgerwood.Temp, legend='Lidgerwood', color='black',    line_width=2)
p1.line(regal.index,      regal.Temp,      legend='Regal',      color='orange',   line_width=2)
p1.line(sheridan.index,   sheridan.Temp,   legend='Sheridan',   color='lightgreen', line_width=2)
p1.line(stevens.index,    stevens.Temp,    legend='Stevens',    color='olive',    line_width=2)
p1.line(EBAM_indoor_overlap.index,  EBAM_indoor_overlap.Temp,  legend='EBAM',       color='gold',     line_width=2)
p1.legend.location='top_left'
tab1 = Panel(child=p1, title="Temperature")


# ....Pressure
p2 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='Pressure (mb)')
p2.line(adams.index,      adams.P,      legend='Adams',      color='blue',     line_width=2)
p2.line(audubon.index,    audubon.P,    legend='Audubon',    color='red',      line_width=2)
p2.line(balboa.index,     balboa.P,     legend='Balboa',     color='green',    line_width=2)
p2.line(browne.index,     browne.P,     legend='Browne',     color='gray',     line_width=2)
p2.line(grant.index,      grant.P,      legend='Grant',      color='cyan',     line_width=2)
p2.line(jefferson.index,  jefferson.P,  legend='Jefferson',  color='magenta',  line_width=2)
p2.line(lidgerwood.index, lidgerwood.P, legend='Lidgerwood', color='black',    line_width=2)
p2.line(regal.index,      regal.P,      legend='Regal',      color='orange',   line_width=2)
p2.line(sheridan.index,   sheridan.P,   legend='Sheridan',   color='lightgreen', line_width=2)
p2.line(stevens.index,    stevens.P,    legend='Stevens',    color='olive',    line_width=2)
p2.line(EBAM_indoor_overlap.index,  EBAM_indoor_overlap.P,  legend='EBAM',       color='gold',     line_width=2)
p2.legend.location='top_left'
tab2 = Panel(child=p2, title="Pressure")

# ....Relative Humidity
p3 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')
p3.line(adams.index,      adams.RH,      legend='Adams',      color='blue',     line_width=2)
p3.line(audubon.index,    audubon.RH,    legend='Audubon',    color='red',      line_width=2)
p3.line(balboa.index,     balboa.RH,     legend='Balboa',     color='green',    line_width=2)
p3.line(browne.index,     browne.RH,     legend='Browne',     color='gray',     line_width=2)
p3.line(grant.index,      grant.RH,      legend='Grant',      color='cyan',     line_width=2)
p3.line(jefferson.index,  jefferson.RH,  legend='Jefferson',  color='magenta',  line_width=2)
p3.line(lidgerwood.index, lidgerwood.RH, legend='Lidgerwood', color='black',    line_width=2)
p3.line(regal.index,      regal.RH,      legend='Regal',      color='orange',   line_width=2)
p3.line(sheridan.index,   sheridan.RH,   legend='Sheridan',   color='lightgreen', line_width=2)
p3.line(stevens.index,    stevens.RH,    legend='Stevens',    color='olive',    line_width=2)
p3.line(EBAM_indoor_overlap.index,  EBAM_indoor_overlap.RH,  legend='EBAM',       color='gold',     line_width=2)
p3.legend.location='top_left'
tab3 = Panel(child=p3, title="Relative Humidity")



# ....PM 10
p4 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 10 (ug m-3)')
p4.line(adams.index,      adams.PM10_standard,      legend='Adams',      color='blue',     line_width=2)
p4.line(audubon.index,    audubon.PM10_standard,    legend='Audubon',    color='red',      line_width=2)
p4.line(balboa.index,     balboa.PM10_standard,     legend='Balboa',     color='green',    line_width=2)
p4.line(browne.index,     browne.PM10_standard,     legend='Browne',     color='gray',     line_width=2)
p4.line(grant.index,      grant.PM10_standard,      legend='Grant',      color='cyan',     line_width=2)
p4.line(jefferson.index,  jefferson.PM10_standard,  legend='Jefferson',  color='magenta',  line_width=2)
p4.line(lidgerwood.index, lidgerwood.PM10_standard, legend='Lidgerwood', color='black',    line_width=2)
p4.line(regal.index,      regal.PM10_standard,      legend='Regal',      color='orange',   line_width=2)
p4.line(sheridan.index,   sheridan.PM10_standard,   legend='Sheridan',   color='lightgreen', line_width=2)
p4.line(stevens.index,    stevens.PM10_standard,    legend='Stevens',    color='olive',    line_width=2)
p4.legend.location='top_left'
tab4 = Panel(child=p4, title="PM 10")

# ....PM 2.5
p5 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM2.5 (ug m-3)')
p5.line(adams.index,      adams.PM2_5_standard,      legend='Adams',      color='blue',     line_width=2)
p5.line(audubon.index,    audubon.PM2_5_standard,    legend='Audubon',    color='red',      line_width=2)
p5.line(balboa.index,     balboa.PM2_5_standard,     legend='Balboa',     color='green',    line_width=2)
p5.line(browne.index,     browne.PM2_5_standard,     legend='Browne',     color='gray',     line_width=2)
p5.line(grant.index,      grant.PM2_5_standard,      legend='Grant',      color='cyan',     line_width=2)
p5.line(jefferson.index,  jefferson.PM2_5_standard,  legend='Jefferson',  color='magenta',  line_width=2)
p5.line(lidgerwood.index, lidgerwood.PM2_5_standard, legend='Lidgerwood', color='black',    line_width=2)
p5.line(regal.index,      regal.PM2_5_standard,      legend='Regal',      color='orange',   line_width=2)
p5.line(sheridan.index,   sheridan.PM2_5_standard,   legend='Sheridan',   color='lightgreen', line_width=2)
p5.line(stevens.index,    stevens.PM2_5_standard,    legend='Stevens',    color='olive',    line_width=2)
p4.line(EBAM_indoor_overlap.index,   EBAM_indoor_overlap.PM2_5,          legend='EBAM',       color='gold',     line_width=2)

p5.legend.location='top_left'
tab5 = Panel(child=p5, title="PM 2.5")

# ....PM 1
p6 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM1 (ug m-3)')
p6.line(adams.index,      adams.PM1_standard,      legend='Adams',      color='blue',     line_width=2)
p6.line(audubon.index,    audubon.PM1_standard,    legend='Audubon',    color='red',      line_width=2)
p6.line(balboa.index,     balboa.PM1_standard,     legend='Balboa',     color='green',    line_width=2)
p6.line(browne.index,     browne.PM1_standard,     legend='Browne',     color='gray',     line_width=2)
p6.line(grant.index,      grant.PM1_standard,      legend='Grant',      color='cyan',     line_width=2)
p6.line(jefferson.index,  jefferson.PM1_standard,  legend='Jefferson',  color='magenta',  line_width=2)
p6.line(lidgerwood.index, lidgerwood.PM1_standard, legend='Lidgerwood', color='black',    line_width=2)
p6.line(regal.index,      regal.PM1_standard,      legend='Regal',      color='orange',   line_width=2)
p6.line(sheridan.index,   sheridan.PM1_standard,   legend='Sheridan',   color='lightgreen', line_width=2)
p6.line(stevens.index,    stevens.PM1_standard,    legend='Stevens',    color='olive',    line_width=2)
p6.legend.location='top_left'
tab6 = Panel(child=p6, title="PM 1")

# ....PM 0.5
p7 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM0.5 (particles/0.1L)')
p7.line(adams.index,      adams.PM_0_5,      legend='Adams',      color='blue',     line_width=2)
p7.line(audubon.index,    audubon.PM_0_5,    legend='Audubon',    color='red',      line_width=2)
p7.line(balboa.index,     balboa.PM_0_5,     legend='Balboa',     color='green',    line_width=2)
p7.line(browne.index,     browne.PM_0_5,     legend='Browne',     color='gray',     line_width=2)
p7.line(grant.index,      grant.PM_0_5,      legend='Grant',      color='cyan',     line_width=2)
p7.line(jefferson.index,  jefferson.PM_0_5,  legend='Jefferson',  color='magenta',  line_width=2)
p7.line(lidgerwood.index, lidgerwood.PM_0_5, legend='Lidgerwood', color='black',    line_width=2)
p7.line(regal.index,      regal.PM_0_5,      legend='Regal',      color='orange',   line_width=2)
p7.line(sheridan.index,   sheridan.PM_0_5,   legend='Sheridan',   color='lightgreen', line_width=2)
p7.line(stevens.index,    stevens.PM_0_5,    legend='Stevens',    color='olive',    line_width=2)
p7.legend.location='top_left'
tab7 = Panel(child=p7, title="PM 0.5")

# ....PM 0.3
p8 = figure(plot_width=900,
            plot_height=450,
            x_range=p1.x_range,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM0.5 (particles/0.1L)')
p8.line(adams.index,      adams.PM_0_3,      legend='Adams',      color='blue',     line_width=2)
p8.line(audubon.index,    audubon.PM_0_3,    legend='Audubon',    color='red',      line_width=2)
p8.line(balboa.index,     balboa.PM_0_3,     legend='Balboa',     color='green',    line_width=2)
p8.line(browne.index,     browne.PM_0_3,     legend='Browne',     color='gray',     line_width=2)
p8.line(grant.index,      grant.PM_0_3,      legend='Grant',      color='cyan',     line_width=2)
p8.line(jefferson.index,  jefferson.PM_0_3,  legend='Jefferson',  color='magenta',  line_width=2)
p8.line(lidgerwood.index, lidgerwood.PM_0_3, legend='Lidgerwood', color='black',    line_width=2)
p8.line(regal.index,      regal.PM_0_3,      legend='Regal',      color='orange',   line_width=2)
p8.line(sheridan.index,   sheridan.PM_0_3,   legend='Sheridan',   color='lightgreen', line_width=2)
p8.line(stevens.index,    stevens.PM_0_3,    legend='Stevens',    color='olive',    line_width=2)
p8.legend.location='top_left'
tab8 = Panel(child=p8, title="PM 0.3")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 ])

show(tabs)
