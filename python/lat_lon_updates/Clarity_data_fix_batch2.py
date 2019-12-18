#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:27:09 2019

@author: matthew
"""

# Obtain Clarity Sensor Data
import requests
import pandas as pd
import datetime
import json
from glob import glob
#%%
#end_time = datetime.datetime.utcnow().isoformat()
end_time = '2019-10-23T20:00:00Z'

start_time = '2019-08-29T07:00:00Z'

limit = '1'

sensor_ID_list = ['A3SFLJL2','AXB8V9BB','AHLRBYVN',
                  'AQ45C95M','AY6GLX7Y']

folder_name_list = ['Sheridan_Sensor_1_A3SFLJL2','Stevens_Sensor_2_AXB8V9BB','Grant_Sensor_4_AHLRBYVN',
                    'Jefferson_Sensor_8_AQ45C95M','Adams_Sensor_9_AY6GLX7Y']


location_name_list = ['Sheridan','Stevens','Grant',
                     'Jefferson','Adams']

solar_data = {}

for name in location_name_list:
    solar_data[name] = pd.DataFrame()

for i in range(len(sensor_ID_list)):
    sensor_ID = sensor_ID_list[i]
    School = location_name_list[i]
    ### Request Sensor Data from Clarity Cloud
    base_url = 'https://clarity-data-api.clarity.io/v1/'
    x_api_key = 'UGQxQRUhFePRybywvdkOdg6gUXN2OvXrHHVpMDpf'
    endpoint = "https://clarity-data-api.clarity.io/v1/measurements"
    headers = {"x-api-key":x_api_key}
#
    # Acquire sensor data based on desired attributes
    data = {"code":sensor_ID,"startTime":start_time,"endTime": end_time,"limit":limit}
    response = requests.get(endpoint,headers=headers,params=data)

    solar_data[School] = response.json()


#%%
from pprint import pprint
pprint(Balboa_backlog[:2])
#%%
#Save Clarity data when using start and end from beginning to 8/22 (sensors on Paccar Roof)

Sheridan_paccar_roof = solar_data['Sheridan']    
Stevens_paccar_roof = solar_data['Stevens']  
Grant_paccar_roof = solar_data['Grant']  
Jefferson_paccar_roof = solar_data['Jefferson']  
Adams_paccar_roof = solar_data['Adams']  

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Paccar_roof_until_8_22/Sheridan_paccar_roof.json','w') as file:
    json.dump(Sheridan_paccar_roof,file)
 
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Paccar_roof_until_8_22/Stevens_paccar_roof.json.json','w') as file:
    json.dump(Stevens_paccar_roof,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Paccar_roof_until_8_22/Grant_paccar_roof.json','w') as file:
    json.dump(Grant_paccar_roof,file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Paccar_roof_until_8_22/Jefferson_paccar_roof.json','w') as file:
    json.dump(Jefferson_paccar_roof,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Paccar_roof_until_8_22/Adams_paccar_roof.json','w') as file:
    json.dump(Adams_paccar_roof,file)



#%%
#Fixing Lat and Lon data
#when using start and end from 8/22 to 8/29 (sensors at facilities building in Spokane)
Sheridan_1 = solar_data['Sheridan']
Stevens_1 = solar_data['Stevens']
Grant_1 = solar_data['Grant']
Jefferson_1 = solar_data['Jefferson']
Adams_1 = solar_data['Adams']

#Spokane School District Facilities location
new = [-117.370646,47.693937]

for value in Sheridan_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Stevens_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Grant_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Jefferson_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Adams_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Facilities_until_8_29/Sheridan_Spokane_Facilities.json','w') as file:
    json.dump(Sheridan_1,file)
 
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Facilities_until_8_29/Stevens_Spokane_Facilities.json','w') as file:
    json.dump(Stevens_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Facilities_until_8_29/Grant_Spokane_Facilities.json','w') as file:
    json.dump(Grant_1,file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Facilities_until_8_29/Jefferson_Spokane_Facilities.json','w') as file:
    json.dump(Jefferson_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Facilities_until_8_29/Adams_Spokane_Facilities.json','w') as file:
    json.dump(Adams_1,file)

#%%
#Fixing lat and lon data
#when using start and end from 8/22 to present (sensors at school locations)
Sheridan_1 = solar_data['Sheridan']
Stevens_1 = solar_data['Stevens']
Grant_1 = solar_data['Grant']
Jefferson_1 = solar_data['Jefferson']
Adams_1 = solar_data['Adams']

#School Latitudes and Longitudes

new_Sheridan = [-117.355561,47.6522472]
new_Stevens = [-117.3846583,47.671256]
new_Grant = [--117.390983,47.6467083]
new_Jefferson = [-117.4098417,47.621533]
new_Adams = [-117.367725,47.621172]

for value in Sheridan_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Sheridan
    
for value in Stevens_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Stevens
    
for value in Grant_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Grant
    
for value in Jefferson_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Jefferson
    
for value in Adams_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Adams
    

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Installed_at_School/Sheridan_School.json','w') as file:
    json.dump(Sheridan_1,file)
 
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Installed_at_School/Stevens_School.json','w') as file:
    json.dump(Stevens_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Installed_at_School/Grant_School.json','w') as file:
    json.dump(Grant_1,file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Installed_at_School/Jefferson_School.json','w') as file:
    json.dump(Jefferson_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Installed_at_School/Adams_Spokane_School.json','w') as file:
    json.dump(Adams_1,file)


#%%
#Combine 3 date dependent Locations into one JSON file for each "Batch 1" sensor   

import itertools

    
result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Sheridan*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Sheridan_backlog = list(itertools.chain.from_iterable(result))
Sheridan_backlog = sorted(Sheridan_backlog, key = lambda i: i['time'])
#Sheridan_backlog = json.dumps(Sheridan_backlog)
#print(Sheridan_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Sheridan_Combined.json", "w") as outfile:
     json.dump(Sheridan_backlog, outfile)
     
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Sheridan_Combined.json') as json_file:
    Sheridan_Combined=json.load(json_file)

result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Stevens*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Stevens_backlog = list(itertools.chain.from_iterable(result))
Stevens_backlog = sorted(Stevens_backlog, key = lambda i: i['time'])
#Stevens_backlog = json.dumps(Stevens_backlog)
#print(Stevens_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Stevens_Combined.json", "w") as outfile:
     json.dump(Stevens_backlog, outfile)
     
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Stevens_Combined.json') as json_file:
    Stevens_Combined = json.load(json_file)
     
result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Grant*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Grant_backlog = list(itertools.chain.from_iterable(result))
Grant_backlog = sorted(Grant_backlog, key = lambda i: i['time'])
#Grant_backlog = json.dumps(Grant_backlog)
#print(Grant_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Grant_Combined.json", "w") as outfile:
     json.dump(Grant_backlog, outfile)
     
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Grant_Combined.json') as json_file:
    Grant_Combined = json.load(json_file)

result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Jefferson*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Jefferson_backlog = list(itertools.chain.from_iterable(result))
Jefferson_backlog = sorted(Jefferson_backlog, key = lambda i: i['time'])
#Jefferson_backlog = json.dumps(Jefferson_backlog)
#print(Jefferson_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Jefferson_Combined.json", "w") as outfile:
     json.dump(Jefferson_backlog, outfile)
     
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Jefferson_Combined.json') as json_file:
    Jefferson_Combined = json.load(json_file)
     
result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Adams*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Adams_backlog = list(itertools.chain.from_iterable(result))
Adams_backlog = sorted(Adams_backlog, key = lambda i: i['time'])
#Adams_backlog = json.dumps(Adams_backlog)
#print(Adams_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Adams_Combined.json", "w") as outfile:
     json.dump(Adams_backlog, outfile)
     
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Adams_Combined.json') as json_file:
    Adams_Combined = json.load(json_file)
    
    
#%%
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Sheridan_Combined.json') as json_file:
    Sheridan_Combined=json.load(json_file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Stevens_Combined.json') as json_file:
    Stevens_Combined = json.load(json_file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Grant_Combined.json') as json_file:
    Grant_Combined = json.load(json_file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Jefferson_Combined.json') as json_file:
    Jefferson_Combined = json.load(json_file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_2/Backlogged_Data_Combined/Adams_Combined.json') as json_file:
    Adams_Combined = json.load(json_file)