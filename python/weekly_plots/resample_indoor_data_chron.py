#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 16:31:40 2022

@author: matthew
"""

#%%
import pandas as pd
from glob import glob
import numpy as np
from datetime import datetime, timedelta
from bokeh.models import Panel, Tabs
from bokeh.layouts import gridplot
#from plot_indoor_out_comparison import indoor_outdoor_plot
#from high_cal_mlr_function_generator import high_cal_setup, generate_mlr_function_high_cal
#from outdoor_low_cal import outdoor_cal_low
from bokeh.io import show
from bokeh.io import export_png
from plot_indoor import plot_indoor
from indoor_correction import indoor_correction

#%%

# Function used to generate a df of datetimes for the previous 7 days for each school 
# (Note that the dates are the same for each school, simply repeated for ease of use in the "one_week_file_compiler" function)

def prev_week_date_generator(previous_dates, site_name):

    previous_days = [1,2,3,4,5,6,7]     
    dates_list = []

    for day in previous_days:

        date = datetime.now() - timedelta(day)
        dates_list.append(date)
        
    previous_dates[site_name] = dates_list

    return previous_dates

    
# Function used to combine the previous 7 days of data into a single df for each indoor sensor

def one_week_file_compiler(previous_dates, site_name, site_df, site_number, bme_site_df):
    
    dates = previous_dates[site_name]
    site_file_list = []
    date_string_list = [] # just used to pull first and last dates for the file names when saving to csv
    
    bme_site_file_list = []
    
    for date in dates:
        
        # load Plantower PM data
        date_string = date.strftime('%Y%m%d')
        file = glob('/Users/matthew/work/PMS_5003_resample_backup/data/ramboll/' + site_name + '/WSU_LAR_Indoor_Air_Quality_Node_' + site_number + '_' + date_string + '*.csv')
       # print(file)
        site_file_list.append(file)
        site_file_list.sort()
        date_string_list.append(date_string)
        
        # load BME sensor data
        file_bme = glob('/Users/matthew/work/PMS_5003_resample_backup/data/ramboll/' + site_name + '/BME_WSU_LAR_Indoor_Air_Quality_Node_' + site_number + '_' + date_string + '*.csv')
      #  print(file_bme)
        bme_site_file_list.append(file_bme)
        bme_site_file_list.sort()
        
    # combine new data files into df for each site (need double "for" loop because "site_file_list" is a list of lists of filename strings
    for file in site_file_list:
        for string in file:
            site_df = pd.concat([site_df, pd.read_csv(string)], sort=False)

    for bme_file in bme_site_file_list:
        for string in bme_file:
            bme_site_df = pd.concat([bme_site_df, pd.read_csv(string)], sort=False)
        
    return site_df, bme_site_df, date_string_list


# Function used to resample and organize raw data

def resample(site, site_name, resample_interval):

    site['Datetime'] = pd.to_datetime(site['Datetime'])
    site = site.sort_values('Datetime')
    place_holder_times = site.Datetime
    del site['Datetime']
    site = site.astype(np.float64)
    site['Datetime'] = place_holder_times
    site.index = site.Datetime
    site = site.resample(resample_interval).mean()
    site['Datetime'] = site.index
    site['Location'] = site_name
    
    return site

# Function used to add BME data to PM dataframe.
# Note that the PM df has one extra row because it has a single data point at 0 seconds on the next day. This is dropped once the BME data is added in 

def combine(Plantower, bme):
    Plantower['PM2_5_corrected'] = Plantower['PM2_5_env']    # this is just renaming so column name matches function input (because for the Clarity nodes, theyd been corrected to the Clarity reference node by this point)
    Plantower['Rel_humid'] = bme['RH']
    Plantower['temp'] = bme['temp']
    Plantower = Plantower[Plantower['temp'].notna()]
    
    return Plantower


#%%

# Set resample interval - used to resample indoor data to lower frequency so that it is more manageable to load in at a future time for analysis.
# Note that data used for analysis is resampled back to 1 hr or 24 hr for analysis after being loaded in as the 15 min resamples created in this script
resample_interval = '15T'
correction_interval = '60T'  

sampling_period = 9 # just used to distinguish which calibration to use for indoor sensors (had a different calibration for smoke event)

# Create empty data frame to hold datetimes for each site
previous_dates = pd.DataFrame({})

# Create list of site names used to generate date string df
# Note that names capitalized because these are going to be used to pull data from the folder where all data is stored (lowercase is for the actual indoor unit dataframes)

site_list = ['Audubon',
             'Adams',
             'Balboa',
             'Browne',
             'Grant',
             'Jefferson',
             'Lidgerwood',
             'Regal',
             'Sheridan',
             'Stevens']

bme_site_list = ['bme_Audubon',
             'bme_Adams',
             'bme_Balboa',
             'bme_Browne',
             'bme_Grant',
             'bme_Jefferson',
             'bme_Lidgerwood',
             'bme_Regal',
             'bme_Sheridan',
             'bme_Stevens']

site_number_list = ['11',
                    '9',
                    '6',
                    '5',
                    '4',
                    '8',
                    '3',
                    '10',
                    '1',
                    '2']
# generate previous week of datetimes for each site

for site_name in site_list:
    
    previous_dates = prev_week_date_generator(previous_dates,site_name)

#%%
    # this cell just testing to see if can condese the following cells using loops (don't use for now)
previous_dates_list = []

for x in range(len(site_list)):
    previous_dates_list.append(previous_dates.iloc[:, 0].tolist())


#df_name_list = ['adams',  'audubon',  'balboa',  'browne',  'grant',  'jefferson', 'lidgerwood',  'regal',  'sheridan',  'stevens', ]
#bme_name_list = ['adams_bme', 'audubon_bme', 'balboa_bme', 'browne_bme', 'grant_bme',  'jefferson_bme', 'lidgerwood_bme', 'regal_bme', 'sheridan_bme', 'stevens_bme']

df_list = {}
bme_df_list = {}

for name in site_list:
        df_list[name] = pd.DataFrame()
        
for name in bme_site_list:
        bme_df_list[name] = pd.DataFrame()

for date, name, df, number, bme in zip(previous_dates_list, site_list, df_list, site_number_list, bme_site_list): 
    
    one_week_file_compiler(date, name, df, number, bme)
        #%%
# create dataframes for each site that have combined the previous 7 days worth of data into a single df
# Note that the date_string list is simply rewritten each time a new locations df is created (it is only needed for the file names when saving to csv)

adams = pd.DataFrame({})
adams_bme = pd.DataFrame({})
adams, adams_bme, date_string_list  = one_week_file_compiler(previous_dates, 'Adams', adams, '9', adams_bme)

audubon = pd.DataFrame({})
audubon_bme = pd.DataFrame({})
audubon, audubon_bme, date_string_list = one_week_file_compiler(previous_dates, 'Audubon', audubon, '11', audubon_bme)

#balboa = pd.DataFrame({})
#balboa_bme = pd.DataFrame({})
#balboa, balboa_bme, date_string_list = one_week_file_compiler(previous_dates, 'Balboa', balboa, '6', balboa_bme)

browne = pd.DataFrame({})
browne_bme = pd.DataFrame({})
browne, browne_bme, date_string_list = one_week_file_compiler(previous_dates, 'Browne', browne, '5', browne_bme)

grant = pd.DataFrame({})
grant_bme = pd.DataFrame({})
grant, grant_bme, date_string_list = one_week_file_compiler(previous_dates, 'Grant', grant, '4', grant_bme)

#jefferson = pd.DataFrame({})
#jefferson_bme = pd.DataFrame({})
#jefferson, jefferson_bme, date_string_list = one_week_file_compiler(previous_dates, 'Jefferson', jefferson, '8', jefferson_bme)

lidgerwood = pd.DataFrame({})
lidgerwood_bme = pd.DataFrame({})
lidgerwood, lidgerwood_bme, date_string_list = one_week_file_compiler(previous_dates, 'Lidgerwood', lidgerwood, '3', lidgerwood_bme)

regal = pd.DataFrame({})
regal_bme = pd.DataFrame({})
regal, regal_bme, date_string_list = one_week_file_compiler(previous_dates, 'Regal', regal, '10', regal_bme)

#sheridan = pd.DataFrame({})
#sheridan_bme = pd.DataFrame({})
#sheridan, sheridan_bme, date_string_list = one_week_file_compiler(previous_dates, 'Sheridan', sheridan, '1', sheridan_bme)

stevens = pd.DataFrame({})
stevens_bme = pd.DataFrame({})
stevens, stevens_bme, date_string_list = one_week_file_compiler(previous_dates, 'Stevens', stevens, '2', stevens_bme)

        

    
#%%

# set date range for resampled file name

date_start = date_string_list[6]
date_end = date_string_list[0]

#%%

# resample indoor data to 15 min frequency
# plantower has one extra data point on next day at 0 seconds, when combine with bme drop the extra line

adams = resample(adams, 'Adams', resample_interval) # doubled checked with other script output, matches
adams_bme = resample(adams_bme, 'Adams', resample_interval)
adams = combine(adams, adams_bme)

audubon = resample(audubon, 'Audubon', resample_interval) # doubled checked with other script output, matches
audubon_bme = resample(audubon_bme, 'Audubon', resample_interval)
audubon = combine(audubon, audubon_bme)

# currently cant connect to balboa

#balboa = resample(balboa, 'Balboa', resample_interval)
#balboa_bme = resample(balboa_bme, 'Balboa', resample_interval)
#balboa = combine(balboa, balboa_bme)

browne = resample(browne, 'Browne', resample_interval) # doubled checked with other script output, matches
browne_bme = resample(browne_bme, 'Browne', resample_interval)
browne = combine(browne, browne_bme)

grant = resample(grant, 'Grant', resample_interval) # doubled checked with other script output, matches
grant_bme = resample(grant_bme, 'Grant', resample_interval)
grant = combine(grant, grant_bme)

# currently Plantower unit at jefferson is down

#jefferson = resample(jefferson, 'Jefferson', resample_interval) # doubled checked with other script output, matches
#jefferson_bme = resample(jefferson_bme, 'Jefferson', resample_interval)
#jefferson = combine(jefferson, jefferson_bme)

lidgerwood = resample(lidgerwood, 'Lidgerwood', resample_interval) # doubled checked with other script output, matches
lidgerwood_bme = resample(lidgerwood_bme, 'Lidgerwood', resample_interval)
lidgerwood = combine(lidgerwood, lidgerwood_bme)

# currently bme at Regal is down 

regal = resample(regal, 'Regal', resample_interval) # doubled checked with other script output, matches
#regal_bme = resample(regal_bme, 'Regal', resample_interval)
#regal = combine(regal, regal_bme)
regal['PM2_5_corrected'] = regal['PM2_5_env']    # manually preform this step for Regal right now because its BME isnt working so cant use "combine" function on it
# currently cant connect to sheridan

#sheridan = resample(sheridan, 'Sheridan', resample_interval)
#sheridan_bme = resample(sheridan_bme, 'Sheridan', resample_interval)
#sheridan = combine(sheridan, sheridan_bme)

stevens = resample(stevens, 'Stevens', resample_interval) # doubled checked with other script output, matches
stevens_bme = resample(stevens_bme, 'Stevens', resample_interval)
stevens = combine(stevens, stevens_bme)

#%%

# save resampled data to csv files (save before resample for plotting to keep 15 min interval because dfs resampled to 1 hour for the correction)

audubon.to_csv('/Users/matthew/work/data/urbanova/ramboll/Audubon/resample_15_min_audubon' + '_' + date_start + '_' + date_end + '.csv', index=False)
adams.to_csv('/Users/matthew/work/data/urbanova/ramboll/Adams/resample_15_min_adams' + '_' + date_start + '_' + date_end + '.csv', index=False)
#balboa.to_csv('/Users/matthew/work/data/urbanova/ramboll/Balboa/resample_15_min_balboa' + '_' + date_start + '_' + date_end + '.csv', index=False)
browne.to_csv('/Users/matthew/work/data/urbanova/ramboll/Browne/resample_15_min_browne' + '_' + date_start + '_' + date_end + '.csv', index=False)
grant.to_csv('/Users/matthew/work/data/urbanova/ramboll/Grant/resample_15_min_grant' + '_' + date_start + '_' + date_end + '.csv', index=False)
#jefferson.to_csv('/Users/matthew/work/data/urbanova/ramboll/Jefferson/resample_15_min_jefferson' + '_' + date_start + '_' + date_end + '.csv', index=False)
lidgerwood.to_csv('/Users/matthew/work/data/urbanova/ramboll/Lidgerwood/resample_15_min_lidgerwood' + '_' + date_start + '_' + date_end + '.csv', index=False)
regal.to_csv('/Users/matthew/work/data/urbanova/ramboll/Regal/resample_15_min_regal' + '_' + date_start + '_' + date_end + '.csv', index=False)
#sheridan.to_csv('/Users/matthew/work/data/urbanova/ramboll/Sheridan/resample_15_min_sheridan' + '_' + date_start + '_' + date_end + '.csv', index=False)
stevens.to_csv('/Users/matthew/work/data/urbanova/ramboll/Stevens/resample_15_stevens' + '_' + date_start + '_' + date_end + '.csv', index=False)


#%%

# apply linear correction to indoor sensors (resamples to 1 hr to apply correction)

audubon = indoor_correction(audubon, 'Audubon', correction_interval)
adams = indoor_correction(adams, 'Adams', correction_interval)
#balboa = indoor_correction(balboa, 'Balboa', correction_interval)
browne = indoor_correction(browne, 'Browne', correction_interval)
grant = indoor_correction(grant, 'Grant', correction_interval)
#jefferson = indoor_correction(jefferson, 'Jefferson', correction_interval)
lidgerwood = indoor_correction(lidgerwood, 'Lidgerwood', correction_interval)
regal = indoor_correction(regal, 'Regal', correction_interval)
#sheridan = indoor_correction(sheridan, 'Sheridan', correction_interval)
stevens = indoor_correction(stevens, 'Stevens', correction_interval)

#%%

# generate plots

p1 = plot_indoor(audubon, 'Audubon')
p2 = plot_indoor(adams, 'Adams')
#p3 = plot_indoor(balboa, 'Balboa')
p4 = plot_indoor(browne, 'Browne')
p5 = plot_indoor(grant, 'Grant')
#p6 = plot_indoor(jefferson, 'Jefferson')
p7 = plot_indoor(lidgerwood, 'Lidgerwood')
p8 = plot_indoor(regal, 'Regal')
#p9 = plot_indoor(sheridan, 'Sheridan') 
p10 = plot_indoor(stevens, 'Stevens') 

p11 = gridplot([[p1,p2], [p4, p5], [p7, p8], [p10]], plot_width = 500, plot_height = 260, toolbar_location=None)

#export_png(p11, filename='/Users/matthew/work/software/urbanova/urbanova-aqnet-rpi-node-school/python/weekly_plots/weekly_plots/' + date_start + '_to_' + date_end + '.png')

tab1 = Panel(child=p11, title="Indoor PM2.5")
tabs = Tabs(tabs=[ tab1])
show(tabs)



