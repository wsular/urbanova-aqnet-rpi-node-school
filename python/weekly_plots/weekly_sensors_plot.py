#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:25:44 2021

@author: matthew
"""

#%%
import pandas as pd
from glob import glob
from bokeh.models import Panel, Tabs
from bokeh.layouts import gridplot
from plot_indoor_out_comparison import indoor_outdoor_plot
from load_indoor_data import load_indoor
from high_cal_mlr_function_generator import high_cal_setup, generate_mlr_function_high_cal
from indoor_cal_low import indoor_cal_low
from outdoor_low_cal import outdoor_cal_low
from bokeh.io import show
from bokeh.io import export_png

#%%

stdev_number = 2   # defines whether using 1 or 2 stdev for uncertainty (one of the options for plotting)

# Date Range of interest
start_time = '2022-04-10 01:00'   
end_time = '2022-04-17 00:00'
sampling_period = '9'
interval = '60T'

weekly_plot_dates = '4_10_to_4_17_22'

#%%


# initiate dataframe for high calibration data used to generate high calibration mlr functions for each location for Clarity Nodes
calibration_df = high_cal_setup()

# generate the mlr for each location based on high calibration data
mlr_high_audubon = generate_mlr_function_high_cal(calibration_df, 'Audubon')
mlr_high_adams = generate_mlr_function_high_cal(calibration_df, 'Adams')
mlr_high_balboa = generate_mlr_function_high_cal(calibration_df, 'Balboa')
mlr_high_browne = generate_mlr_function_high_cal(calibration_df, 'Browne')
mlr_high_grant = generate_mlr_function_high_cal(calibration_df, 'Grant')
mlr_high_jefferson = generate_mlr_function_high_cal(calibration_df, 'Jefferson')
mlr_high_lidgerwood = generate_mlr_function_high_cal(calibration_df, 'Lidgerwood')
mlr_high_regal = generate_mlr_function_high_cal(calibration_df, 'Regal')
mlr_high_sheridan = generate_mlr_function_high_cal(calibration_df, 'Sheridan')
mlr_high_stevens = generate_mlr_function_high_cal(calibration_df, 'Stevens')

#%%

#Import entire data set of Clarity Data for each location

Audubon_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Audubon*.csv')
files.sort()
for file in files:
    Audubon_All = pd.concat([Audubon_All, pd.read_csv(file)], sort=False)


Adams_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Adams*.csv')
files.sort()
for file in files:
    Adams_All = pd.concat([Adams_All, pd.read_csv(file)], sort=False)



Balboa_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Balboa*.csv')
files.sort()
for file in files:
    Balboa_All = pd.concat([Balboa_All, pd.read_csv(file)], sort=False)



Browne_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Browne*.csv')
files.sort()
for file in files:
    Browne_All = pd.concat([Browne_All, pd.read_csv(file)], sort=False)


# drop erroneous data from Nov. 2019 when sensor malfunctioning
Browne_All = Browne_All[Browne_All['PM2_5'] < 1000]


Grant_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Grant*.csv')
files.sort()
for file in files:
    Grant_All = pd.concat([Grant_All, pd.read_csv(file)], sort=False)



Jefferson_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Jefferson*.csv')
files.sort()
for file in files:
    Jefferson_All = pd.concat([Jefferson_All, pd.read_csv(file)], sort=False)



Lidgerwood_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Lidgerwood*.csv')
files.sort()
for file in files:
    Lidgerwood_All = pd.concat([Lidgerwood_All, pd.read_csv(file)], sort=False)



Regal_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Regal*.csv')
files.sort()
for file in files:
    Regal_All = pd.concat([Regal_All, pd.read_csv(file)], sort=False)


    
Sheridan_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Sheridan*.csv')
files.sort()
for file in files:
    Sheridan_All = pd.concat([Sheridan_All, pd.read_csv(file)], sort=False)



Stevens_All = pd.DataFrame({})
files = glob('/Users/matthew/work/data/Clarity_Backup/Stevens*.csv')
files.sort()
for file in files:
    Stevens_All = pd.concat([Stevens_All, pd.read_csv(file)], sort=False)



# resample and cut Clarity data to time period of interest

Audubon_All['time'] = pd.to_datetime(Audubon_All['time'])
Audubon_All = Audubon_All.sort_values('time')
Audubon_All.index = Audubon_All.time
Audubon = Audubon_All.loc[start_time:end_time]

Adams_All['time'] = pd.to_datetime(Adams_All['time'])
Adams_All = Adams_All.sort_values('time')
Adams_All.index = Adams_All.time
Adams = Adams_All.loc[start_time:end_time]


Balboa_All['time'] = pd.to_datetime(Balboa_All['time'])
Balboa_All = Balboa_All.sort_values('time')
Balboa_All.index = Balboa_All.time
Balboa = Balboa_All.loc[start_time:end_time]  


Browne_All['time'] = pd.to_datetime(Browne_All['time'])
Browne_All = Browne_All.sort_values('time')
Browne_All.index = Browne_All.time
Browne = Browne_All.loc[start_time:end_time]

# needed to also drop the data from when node working again (in Paccar lab) to when it was re-installed in Spokane
Browne_start_1 = start_time
Browne_end_1 = '2019-11-15 00:00'

Browne_start_2 = '2019-11-25 00:00'
Browne_end_2 = end_time

Browne_1 = Browne.loc[Browne_start_1:Browne_end_1]
Browne_2 = Browne.loc[Browne_start_2:Browne_end_2]

Browne = Browne_1.append(Browne_2)
Browne = Browne.sort_index()



Grant_All['time'] = pd.to_datetime(Grant_All['time'])
Grant_All = Grant_All.sort_values('time')
Grant_All.index = Grant_All.time
Grant = Grant_All.loc[start_time:end_time]

Grant_count = Grant.groupby(Grant.index.date).count()

Jefferson_All['time'] = pd.to_datetime(Jefferson_All['time'])
Jefferson_All = Jefferson_All.sort_values('time')
Jefferson_All.index = Jefferson_All.time
Jefferson = Jefferson_All.loc[start_time:end_time]


Lidgerwood_All['time'] = pd.to_datetime(Lidgerwood_All['time'])
Lidgerwood_All = Lidgerwood_All.sort_values('time')
Lidgerwood_All.index = Lidgerwood_All.time
Lidgerwood = Lidgerwood_All.loc[start_time:end_time]


Regal_All['time'] = pd.to_datetime(Regal_All['time'])
Regal_All = Regal_All.sort_values('time')
Regal_All.index = Regal_All.time
Regal = Regal_All.loc[start_time:end_time]


Sheridan_All['time'] = pd.to_datetime(Sheridan_All['time'])
Sheridan_All = Sheridan_All.sort_values('time')
Sheridan_All.index = Sheridan_All.time
Sheridan = Sheridan_All.loc[start_time:end_time]


Stevens_All['time'] = pd.to_datetime(Stevens_All['time'])
Stevens_All = Stevens_All.sort_values('time')
Stevens_All.index = Stevens_All.time
Stevens = Stevens_All.loc[start_time:end_time]

# resample to interval used to generate the corrections

Audubon = Audubon.resample(interval).mean() 
Adams = Adams.resample(interval).mean()  
Balboa = Balboa.resample(interval).mean()
Browne = Browne.resample(interval).mean()
Grant = Grant.resample(interval).mean()
Jefferson = Jefferson.resample(interval).mean()
Lidgerwood = Lidgerwood.resample(interval).mean()
Regal = Regal.resample(interval).mean()
Sheridan = Sheridan.resample(interval).mean()
Stevens = Stevens.resample(interval).mean()


Audubon = Audubon.dropna()
Adams = Adams.dropna()
Balboa = Balboa.dropna()
Browne = Browne.dropna()
Grant = Grant.dropna()
Jefferson = Jefferson.dropna()
Lidgerwood = Lidgerwood.dropna()
Regal = Regal.dropna()
Sheridan = Sheridan.dropna()
Stevens = Stevens.dropna()

Audubon['Location'] = 'Audubon'
Adams['Location'] = 'Adams'
Balboa['Location'] = 'Balboa'
Browne['Location'] = 'Browne'
Grant['Location'] = 'Grant'
Jefferson['Location'] = 'Jefferson'
Lidgerwood['Location'] = 'Lidgerwood'
Regal['Location'] = 'Regal'
Sheridan['Location'] = 'Sheridan'

#%%

Audubon_low = outdoor_cal_low(Audubon, 'Audubon', time_period = sampling_period)
Adams_low = outdoor_cal_low(Adams, 'Adams', time_period = sampling_period)
Balboa_low = outdoor_cal_low(Balboa, 'Balboa', time_period = sampling_period)
Browne_low = outdoor_cal_low(Browne, 'Browne', time_period = sampling_period)
Grant_low = outdoor_cal_low(Grant, 'Grant', time_period = sampling_period)
Jefferson_low = outdoor_cal_low(Jefferson, 'Jefferson', time_period = sampling_period)
Lidgerwood_low= outdoor_cal_low(Lidgerwood, 'Lidgerwood', time_period = sampling_period)
Regal_low =  outdoor_cal_low(Regal, 'Regal', time_period = sampling_period)
Sheridan_low = outdoor_cal_low(Sheridan, 'Sheridan', time_period = sampling_period)
Stevens_low = outdoor_cal_low(Stevens, 'Stevens', time_period = sampling_period)

# use this when not including the sept 2020 smoke period
Audubon = Audubon_low
Adams = Adams_low
Balboa = Balboa_low
Browne = Browne_low
Grant = Grant_low
Jefferson = Jefferson_low
Lidgerwood = Lidgerwood_low
Regal = Regal_low
Sheridan = Sheridan_low
Stevens = Stevens_low

Audubon = Audubon.sort_index()
Adams = Adams.sort_index()
Balboa = Balboa.sort_index()
Browne = Browne.sort_index()
Grant = Grant.sort_index()
Jefferson = Jefferson.sort_index()
Lidgerwood = Lidgerwood.sort_index()
Regal = Regal.sort_index()
Sheridan = Sheridan.sort_index()
Stevens = Stevens.sort_index()

#%%
audubon_bme = pd.DataFrame({})
audubon_bme_json = pd.DataFrame({})

adams_bme = pd.DataFrame({})
adams_bme_json = pd.DataFrame({})

balboa_bme = pd.DataFrame({})
balboa_bme_json = pd.DataFrame({})

browne_bme = pd.DataFrame({})
browne_bme_json = pd.DataFrame({})

grant_bme = pd.DataFrame({})
grant_bme_json = pd.DataFrame({})

jefferson_bme = pd.DataFrame({})
jefferson_bme_json = pd.DataFrame({})

lidgerwood_bme = pd.DataFrame({})
lidgerwood_bme_json = pd.DataFrame({})

regal_bme = pd.DataFrame({})
regal_bme_json = pd.DataFrame({})

sheridan_bme = pd.DataFrame({})
sheridan_bme_json = pd.DataFrame({})

stevens_bme = pd.DataFrame({})
stevens_bme_json = pd.DataFrame({})


audubon_bme, audubon_bme_json = load_indoor('Audubon', audubon_bme,audubon_bme_json, interval,
                                                time_period_4 = 'no', start = start_time, stop = end_time)


adams_bme, adams_bme_json = load_indoor('Adams', adams_bme,adams_bme_json, interval,
                                            time_period_4 = 'no', start = start_time, stop = end_time)

balboa_bme, balboa_bme_json = load_indoor('Balboa', balboa_bme,balboa_bme_json, interval,
                                              time_period_4 = 'no', start = start_time, stop = end_time)

browne_bme, browne_bme_json = load_indoor('Browne', browne_bme,browne_bme_json, interval,
                                              time_period_4 = 'no', start = start_time, stop = end_time)

grant_bme, grant_bme_json = load_indoor('Grant', grant_bme,grant_bme_json, interval,
                                            time_period_4 = 'no', start = start_time, stop = end_time)

jefferson_bme, jefferson_bme_json = load_indoor('Jefferson', jefferson_bme,jefferson_bme_json, interval,
                                                    time_period_4 = 'no', start = start_time, stop = end_time)


lidgerwood_bme, lidgerwood_bme_json = load_indoor('Lidgerwood', lidgerwood_bme,lidgerwood_bme_json, interval,
                                                      time_period_4 = 'no', start = start_time, stop = end_time)

regal_bme, regal_bme_json = load_indoor('Regal', regal_bme,regal_bme_json, interval,
                                            time_period_4 = 'no', start = start_time, stop = end_time)


#sheridan_bme, sheridan_bme_json = load_indoor('Sheridan', sheridan_bme,sheridan_bme_json, interval,
#                                                  time_period_4 = 'no', start = start_time, stop = end_time)


stevens_bme, stevens_bme_json = load_indoor('Stevens', stevens_bme,stevens_bme_json, interval,
                                                time_period_4 = 'no', start = start_time, stop = end_time)
    
    
grant = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Grant/resample*.csv')
files.sort()
for file in files:
    grant = pd.concat([grant, pd.read_csv(file)], sort=False)

sheridan = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Sheridan/resample*.csv')
files.sort()
for file in files:
    sheridan = pd.concat([sheridan, pd.read_csv(file)], sort=False)
    
adams = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Adams/resample*.csv')
files.sort()
for file in files:
    adams = pd.concat([adams, pd.read_csv(file)], sort=False)

stevens = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Stevens/resample*.csv')
files.sort()
for file in files:
    stevens = pd.concat([stevens, pd.read_csv(file)], sort=False)


balboa = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Balboa/resample*.csv')
files.sort()
for file in files:
    balboa = pd.concat([balboa, pd.read_csv(file)], sort=False)


jefferson = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Jefferson/resample*.csv')
files.sort()
for file in files:
    jefferson = pd.concat([jefferson, pd.read_csv(file)], sort=False)


browne = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Browne/resample*.csv')
files.sort()
for file in files:
    browne = pd.concat([browne, pd.read_csv(file)], sort=False)


audubon = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Audubon/resample*.csv')
files.sort()
for file in files:
    audubon = pd.concat([audubon, pd.read_csv(file)], sort=False)


lidgerwood = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Lidgerwood/resample*.csv')
files.sort()
for file in files:
    lidgerwood = pd.concat([lidgerwood, pd.read_csv(file)], sort=False)


regal = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Regal/resample*.csv')
files.sort()
for file in files:
    regal = pd.concat([regal, pd.read_csv(file)], sort=False)
    
    
    
# add temp and relative humidity columns from BME280 to indoor unit PM 2.5 df
adams['Datetime'] = pd.to_datetime(adams['Datetime'])
adams.index = adams.Datetime
audubon['Datetime'] = pd.to_datetime(audubon['Datetime'])
audubon.index = audubon.Datetime
balboa['Datetime'] = pd.to_datetime(balboa['Datetime'])
balboa.index = balboa.Datetime
browne['Datetime'] = pd.to_datetime(browne['Datetime'])
browne.index = browne.Datetime
grant['Datetime'] = pd.to_datetime(grant['Datetime'])
grant.index = grant.Datetime
jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson.index = jefferson.Datetime
lidgerwood['Datetime'] = pd.to_datetime(lidgerwood['Datetime'])
lidgerwood.index = lidgerwood.Datetime
regal['Datetime'] = pd.to_datetime(regal['Datetime'])
regal.index = regal.Datetime
#sheridan['Datetime'] = pd.to_datetime(sheridan['Datetime'])
#sheridan.index = sheridan.Datetime
stevens['Datetime'] = pd.to_datetime(stevens['Datetime'])
stevens.index = stevens.Datetime
#%%

adams = adams.loc[start_time:end_time]
audubon = audubon.loc[start_time:end_time]
balboa = balboa.loc[start_time:end_time]
browne = browne.loc[start_time:end_time]
grant = grant.loc[start_time:end_time]
jefferson = jefferson.loc[start_time:end_time]
lidgerwood = lidgerwood.loc[start_time:end_time]
regal = regal.loc[start_time:end_time]
# sheridan = sheridan.loc[start_time:end_time]
stevens = stevens.loc[start_time:end_time]


adams = adams.resample(interval).mean() 
adams['PM2_5_corrected'] = adams['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
adams['Rel_humid'] = adams_bme['RH']
adams['temp'] = adams_bme['temp']

audubon = audubon.resample(interval).mean() 
audubon['PM2_5_corrected'] = audubon['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
audubon['Rel_humid'] = audubon_bme['RH']
audubon['temp'] = audubon_bme['temp']

balboa = balboa.resample(interval).mean() 
balboa['PM2_5_corrected'] = balboa['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
balboa['Rel_humid'] = balboa_bme['RH']
balboa['temp'] = balboa_bme['temp']

browne = browne.resample(interval).mean() 
browne['PM2_5_corrected'] = browne['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
browne['Rel_humid'] = browne_bme['RH']
browne['temp'] = browne_bme['temp']

grant = grant.resample(interval).mean() 
grant['PM2_5_corrected'] = grant['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
grant['Rel_humid'] = grant_bme['RH']
grant['temp'] = grant_bme['temp']

jefferson = jefferson.resample(interval).mean() 
jefferson['PM2_5_corrected'] = jefferson['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
jefferson['Rel_humid'] = jefferson_bme['RH']
jefferson['temp'] = jefferson_bme['temp']

lidgerwood = lidgerwood.resample(interval).mean() 
lidgerwood['PM2_5_corrected'] = lidgerwood['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
lidgerwood['Rel_humid'] = lidgerwood_bme['RH']
lidgerwood['temp'] = lidgerwood_bme['temp']

regal = regal.resample(interval).mean() 
regal['PM2_5_corrected'] = regal['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
regal['Rel_humid'] = regal_bme['RH']
regal['temp'] = regal_bme['temp']

#sheridan = sheridan.resample(interval).mean() 
#sheridan['PM2_5_corrected'] = sheridan['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
#sheridan['Rel_humid'] = sheridan_bme['RH']
#sheridan['temp'] = sheridan_bme['temp']

stevens = stevens.resample(interval).mean() 
stevens['PM2_5_corrected'] = stevens['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
stevens['Rel_humid'] = stevens_bme['RH']
stevens['temp'] = stevens_bme['temp']

audubon['Location'] = 'Audubon'
adams['Location'] = 'Adams'
balboa['Location'] = 'Balboa'
browne['Location'] = 'Browne'
grant['Location'] = 'Grant'
jefferson['Location'] = 'Jefferson'
lidgerwood['Location'] = 'Lidgerwood'
regal['Location'] = 'Regal'
#sheridan['Location'] = 'Sheridan'
stevens['Location'] = 'Stevens'
#%%
# use when not in smoke cal times

audubon_low = indoor_cal_low(audubon, 'Audubon', sampling_period)
adams_low = indoor_cal_low(adams, 'Adams', sampling_period)
balboa_low = indoor_cal_low(balboa, 'Balboa', sampling_period)
browne_low = indoor_cal_low(browne, 'Browne', sampling_period)
grant_low = indoor_cal_low(grant, 'Grant', sampling_period)
jefferson_low = indoor_cal_low(jefferson, 'Jefferson', sampling_period)
lidgerwood_low = indoor_cal_low(lidgerwood, 'Lidgerwood', sampling_period)
regal_low = indoor_cal_low(regal, 'Regal', sampling_period)
#sheridan_low = indoor_cal_low(sheridan, 'Sheridan', sampling_period)
stevens_low = indoor_cal_low(stevens, 'Stevens', sampling_period)

audubon = audubon_low
adams = adams_low
balboa = balboa_low
browne = browne_low
grant = grant_low
jefferson = jefferson_low
lidgerwood = lidgerwood_low
regal = regal_low
#sheridan = sheridan_low
stevens = stevens_low


audubon = audubon.sort_index()
adams = adams.sort_index()
balboa = balboa.sort_index()
browne = browne.sort_index()
grant = grant.sort_index()
jefferson = jefferson.sort_index()
lidgerwood = lidgerwood.sort_index()
regal = regal.sort_index()
#sheridan = sheridan.sort_index()
stevens = stevens.sort_index()

#%%
# input arg 3 is the plot title, arg 4 is the in/out compare time period (for saving png's of figures to correct folder) make sure to change based on the selected time period so don't overwrite figures
p1 = indoor_outdoor_plot(audubon, Audubon, 'Audubon', stdev_number,sampling_period, shift = 'no') # Site 1

p2 = indoor_outdoor_plot(adams, Adams, 'Adams', stdev_number, sampling_period, shift = 'no') # Site 2

#p3 = indoor_outdoor_plot(balboa, Balboa, 'Balboa', stdev_number, sampling_period, shift = 'no') # Site 3

p4 = indoor_outdoor_plot(browne, Browne, 'Browne', stdev_number, sampling_period, shift = 'no') # Site 4

p5 = indoor_outdoor_plot(grant, Grant, 'Grant', stdev_number, sampling_period, shift = 'no') # Site 5

p6 = indoor_outdoor_plot(jefferson, Jefferson, 'Jefferson', stdev_number, sampling_period, shift = 'no') # Site 6

p7 = indoor_outdoor_plot(lidgerwood, Lidgerwood, 'Lidgerwood', stdev_number, sampling_period, shift = 'no') # Site 7

p8 = indoor_outdoor_plot(regal, Regal, 'Regal', stdev_number, sampling_period, shift = 'no') # Site 8

#p9 = indoor_outdoor_plot(sheridan, Sheridan, 'Sheridan', stdev_number, sampling_period, shift = 'no') # Site 9

p10 = indoor_outdoor_plot(stevens, Stevens, 'Stevens', stdev_number, sampling_period, shift = 'no') # Site 10


# weekly plot dates


#p11 = gridplot([[p1,p2], [p3, p4], [p5, p6], [p7, p8], [p9, p10]], plot_width = 500, plot_height = 260, toolbar_location=None)
p11 = gridplot([[p1,p2], [p4, p5], [p6, p7], [p8, p10]], plot_width = 500, plot_height = 260, toolbar_location=None)
#p11 = gridplot([[p4,p7], [p8, p9], [p10]], plot_width = 500, plot_height = 260, toolbar_location=None)  # for plotting ATP 5 where it is just the 5 schools where both units functioning during smoke event

# make sure to change the In_out_compare number to put in correct folder based on the time period

#export_png(p11, filename='/Users/matthew/Desktop/thesis/Final_Figures/In_out_compare_' + sampling_period + '/all_sites_gridplot_unshifted.png')
export_png(p11, filename='/Users/matthew/work/software/urbanova/urbanova-aqnet-rpi-node-school/python/weekly_plots/weekly_plots/' + weekly_plot_dates + '.png')

tab1 = Panel(child=p11, title="Indoor Outdoor Comparison")

tabs = Tabs(tabs=[ tab1])

show(tabs)