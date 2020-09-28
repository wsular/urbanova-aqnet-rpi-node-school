#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 19:37:02 2020

@author: matthew
"""
import json
import requests
import pandas as pd
import datetime
from glob import glob
#%%

# retrieve data from Augusta site for PM 2.5 from Dec 18, 2019 to Dec 31, 2019 (request has to be within the same year)
#response = requests.get("https://aqs.epa.gov/data/api/sampleData/bySite?email=matthew.s.roetcisoe@wsu.edu&key=greyfox35&param=88101&bdate=20191218&edate=20191231&state=53&county=063&site=0021")

# retrieve data from Augusta site for PM 2.5 from Jan 1 to March 5 2020 (request has to be within the same year)
response = requests.get("https://aqs.epa.gov/data/api/sampleData/bySite?email=matthew.s.roetcisoe@wsu.edu&key=greyfox35&param=88101&bdate=20200101&edate=20200105&state=53&county=063&site=0021")


json_file = response.json()
#%%
json_file = json_file['Data']

# For 2019 data
#with open('/Users/matthew/Desktop/data/AirNow/Augusta_2019.json', 'w', encoding='utf-8') as f:
#   json.dump(json_file, f, ensure_ascii=False, indent=4)

# For 2020 data
with open('/Users/matthew/Desktop/data/AirNow/Augusta_2020.json', 'w', encoding='utf-8') as f:
    json.dump(json_file, f, ensure_ascii=False, indent=4)
#%%


Augusta = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/AirNow/Augusta*.json')
files.sort()
for file in files:
    Augusta = pd.concat([Augusta, pd.read_json(file)], sort=False)

Augusta_save = pd.DataFrame({})
Augusta_save['time'] = Augusta['date_local']
Augusta_save['PM2_5'] = Augusta['sample_measurement']

# save 2019 data
#Augusta_save.to_csv('/Users/matthew/Desktop/data/AirNow/Augusta_2019.csv', index=False)    # save for Augusta Dec 18 2019 to Dec 31 2019

# save 2020 data
Augusta_save.to_csv('/Users/matthew/Desktop/data/AirNow/Augusta_2020.csv', index=False)    # save for Jan 1 - March 5 2020

#%%

Augusta = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/AirNow/Augusta*.csv')
files.sort()
for file in files:
    Augusta = pd.concat([Augusta, pd.read_csv(file)], sort=False)
Augusta = Augusta.sort_values('time')
Augusta.index = Augusta.time