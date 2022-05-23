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

#%%

# Date Range to be resampled

start_time = '2022-05-03 01:00'   
end_time = '2022-05-09 00:00'
sampling_period = '9'

interval = '15T'  # only used for resampling indoor data so more manageable and doesnt take 20 min to load in... use 1 hr and 24 hr for any analysis as these are what the calibrations are based on 


#%%

# read in data for indoor units from "/Users/matthew/work/data/urbanova/ramboll/" folder
# Note that only the most recent data is available in "/Users/matthew/work/data/urbanova/ramboll/" folder. This data has been manually copied and pasted from the
# "/Users/matthew/Desktop/PMS_5003_resample_backup/data/ramboll" folder to which data is rsynced from Gaia (all data is found in /Users/matthew/Desktop/PMS_5003_resample_backup/data/ramboll)
# After the most current data has been resampled in "/Users/matthew/work/data/urbanova/ramboll/", it is removed (to be manually replaced with the 
# newest data from /Users/matthew/Desktop/PMS_5003_resample_backup/data/ramboll after the next rsync)


grant = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Grant/WSU*.csv')
files.sort()
for file in files:
    grant = pd.concat([grant, pd.read_csv(file)], sort=False)

sheridan = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Sheridan/WSU*.csv')
files.sort()
for file in files:
    sheridan = pd.concat([sheridan, pd.read_csv(file)], sort=False)
    
adams = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Adams/WSU*.csv')
files.sort()
for file in files:
    adams = pd.concat([adams, pd.read_csv(file)], sort=False)

stevens = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Stevens/WSU*.csv')
files.sort()
for file in files:
    stevens = pd.concat([stevens, pd.read_csv(file)], sort=False)


balboa = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Balboa/WSU*.csv')
files.sort()
for file in files:
    balboa = pd.concat([balboa, pd.read_csv(file)], sort=False)


jefferson = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Jefferson/WSU*.csv')
files.sort()
for file in files:
    jefferson = pd.concat([jefferson, pd.read_csv(file)], sort=False)


browne = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Browne/WSU*.csv')
files.sort()
for file in files:
    browne = pd.concat([browne, pd.read_csv(file)], sort=False)


audubon = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Audubon/WSU*.csv')
files.sort()
for file in files:
    audubon = pd.concat([audubon, pd.read_csv(file)], sort=False)


lidgerwood = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Lidgerwood/WSU*.csv')
files.sort()
for file in files:
    lidgerwood = pd.concat([lidgerwood, pd.read_csv(file)], sort=False)


regal = pd.DataFrame({})
files   = glob('/Users/matthew/work/data/urbanova/ramboll/Regal/WSU*.csv')
files.sort()
for file in files:
    regal = pd.concat([regal, pd.read_csv(file)], sort=False)
    
#%%
# used to resample indoor data to lower frequency so doesnt take so long to load in each time

grant['Datetime'] = pd.to_datetime(grant['Datetime'])
grant = grant.sort_values('Datetime')

place_holder_times = grant.Datetime
del grant['Datetime']
grant = grant.astype(np.float64)
grant['Datetime'] = place_holder_times

grant.index = grant.Datetime
grant = grant.loc[start_time:end_time]
grant = grant.resample(interval).mean()
grant['Datetime'] = grant.index
grant['Location'] = 'Grant'


stevens['Datetime'] = pd.to_datetime(stevens['Datetime'])
stevens = stevens.sort_values('Datetime')

place_holder_times = stevens.Datetime
del stevens['Datetime']
stevens = stevens.astype(np.float64)
stevens['Datetime'] = place_holder_times

stevens.index = stevens.Datetime
stevens = stevens.loc[start_time:end_time] 
stevens = stevens.resample(interval).mean()
stevens['Datetime'] = stevens.index
stevens['Location'] = 'Stevens'


balboa['Datetime'] = pd.to_datetime(balboa['Datetime'])
balboa = balboa.sort_values('Datetime')

place_holder_times = balboa.Datetime
del balboa['Datetime']
balboa = balboa.astype(np.float64)
balboa['Datetime'] = place_holder_times

balboa.index = balboa.Datetime
balboa = balboa.loc[start_time:end_time] 
balboa = balboa.resample(interval).mean()
balboa['Datetime'] = balboa.index
balboa['Location'] = 'Balboa'


adams['Datetime'] = pd.to_datetime(adams['Datetime'])
adams = adams.sort_values('Datetime')

place_holder_times = adams.Datetime
del adams['Datetime']
adams = adams.astype(np.float64)
adams['Datetime'] = place_holder_times

adams.index = adams.Datetime
adams = adams.loc[start_time:end_time] 
adams = adams.resample(interval).mean()
adams['Datetime'] = adams.index
adams['Location'] = 'Adams'


jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson = jefferson.sort_values('Datetime')

place_holder_times = jefferson.Datetime
del jefferson['Datetime']
jefferson = jefferson.astype(np.float64)
jefferson['Datetime'] = place_holder_times

jefferson.index = jefferson.Datetime
jefferson = jefferson.loc[start_time:end_time] 
jefferson = jefferson.resample(interval).mean()
jefferson['Datetime'] = jefferson.index
jefferson['Location'] = 'Jefferson'


sheridan['Datetime'] = pd.to_datetime(sheridan['Datetime'])
sheridan = sheridan.sort_values('Datetime')

place_holder_times = sheridan.Datetime
del sheridan['Datetime']

sheridan = sheridan.astype(np.float64)
sheridan['Datetime'] = place_holder_times

sheridan.index = sheridan.Datetime
sheridan = sheridan.loc[start_time:end_time] 
sheridan = sheridan.resample(interval).mean()
sheridan['Datetime'] = sheridan.index
sheridan['Location'] = 'Sheridan'


browne['Datetime'] = pd.to_datetime(browne['Datetime'])
browne = browne.sort_values('Datetime')

place_holder_times = browne.Datetime
del browne['Datetime']
browne = browne.astype(np.float64)
browne['Datetime'] = place_holder_times

browne.index = browne.Datetime
browne = browne.loc[start_time:end_time] 
browne = browne.resample(interval).mean()
browne['Datetime'] = browne.index
browne['Location'] = 'Browne'


audubon['Datetime'] = pd.to_datetime(audubon['Datetime'])
audubon = audubon.sort_values('Datetime')

place_holder_times = audubon.Datetime
del audubon['Datetime']
audubon = audubon.astype(np.float64)
audubon['Datetime'] = place_holder_times

audubon.index = audubon.Datetime
audubon = audubon.loc[start_time:end_time] 
audubon = audubon.resample(interval).mean()
audubon['Datetime'] = audubon.index
audubon['Location'] = 'Audubon'


lidgerwood['Datetime'] = pd.to_datetime(lidgerwood['Datetime'])
lidgerwood = lidgerwood.sort_values('Datetime')

place_holder_times = lidgerwood.Datetime
del lidgerwood['Datetime']
lidgerwood = lidgerwood.astype(np.float64)
lidgerwood['Datetime'] = place_holder_times

lidgerwood.index = lidgerwood.Datetime
lidgerwood = lidgerwood.loc[start_time:end_time] 
lidgerwood = lidgerwood.resample(interval).mean()
lidgerwood['Datetime'] = lidgerwood.index
lidgerwood['Location'] = 'Lidgerwood'


regal['Datetime'] = pd.to_datetime(regal['Datetime'])
regal = regal.sort_values('Datetime')

place_holder_times = regal.Datetime
del regal['Datetime']
regal = regal.astype(np.float64)
regal['Datetime'] = place_holder_times

regal.index = regal.Datetime
regal = regal.loc[start_time:end_time] 
regal = regal.resample(interval).mean()
regal['Datetime'] = regal.index
regal['Location'] = 'Regal'

#%%

# set date range for resampled file name

date_range = '5_03_to_5_09_22'
#%%

# save resampled data to csv files

audubon.to_csv('/Users/matthew/work/data/urbanova/ramboll/Audubon/resample_15_min_audubon' + '_' + date_range + '.csv', index=False)
adams.to_csv('/Users/matthew/work/data/urbanova/ramboll/Adams/resample_15_min_adams' + '_' + date_range + '.csv', index=False)
balboa.to_csv('/Users/matthew/work/data/urbanova/ramboll/Balboa/resample_15_min_balboa' + '_' + date_range + '.csv', index=False)
browne.to_csv('/Users/matthew/work/data/urbanova/ramboll/Browne/resample_15_min_browne' + '_' + date_range + '.csv', index=False)
grant.to_csv('/Users/matthew/work/data/urbanova/ramboll/Grant/resample_15_min_grant' + '_' + date_range + '.csv', index=False)
jefferson.to_csv('/Users/matthew/work/data/urbanova/ramboll/Jefferson/resample_15_min_jefferson' + '_' + date_range + '.csv', index=False)
lidgerwood.to_csv('/Users/matthew/work/data/urbanova/ramboll/Lidgerwood/resample_15_min_lidgerwood' + '_' + date_range + '.csv', index=False)
regal.to_csv('/Users/matthew/work/data/urbanova/ramboll/Regal/resample_15_min_regal' + '_' + date_range + '.csv', index=False)
sheridan.to_csv('/Users/matthew/work/data/urbanova/ramboll/Sheridan/resample_15_min_sheridan' + '_' + date_range + '.csv', index=False)
stevens.to_csv('/Users/matthew/work/data/urbanova/ramboll/Stevens/resample_15_stevens' + '_' + date_range + '.csv', index=False)
