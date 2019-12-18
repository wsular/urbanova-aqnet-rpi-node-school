#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:46:39 2019

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
end_time = '2019-09-01T20:00:00Z'

start_time = '2019-08-22T07:00:00Z'

limit = '5000'

sensor_ID_list = ['A3SFLJL2','AXB8V9BB','A5Y0CDHY','AHLRBYVN','ABT9RVX9','A8XWLKZZ',
                  'AQ45C95M','AY6GLX7Y','ADTRSXZJ','AT5TF0BK','AKYWSPV0']

folder_name_list = ['Sheridan_Sensor_1_A3SFLJL2','Stevens_Sensor_2_AXB8V9BB','Lidgerwood_Sensor_3_A5Y0CDHY','Grant_Sensor_4_AHLRBYVN','Browne_Sensor_5_ABT9RVX9',
                    'Balboa_Sensor_6_A8XWLKZZ','Jefferson_Sensor_8_AQ45C95M','Adams_Sensor_9_AY6GLX7Y','Regal_Sensor_10_ADTRSXZJ','Audubon_Sensor_11_AT5TF0BK']


location_name_list = ['Sheridan','Stevens','Lidgerwood','Grant','Browne',
                    'Balboa','Jefferson','Adams','Regal','Audubon','Reference_site']


sensor_number_list = ['1','2','3','4','5','6','8','9','10','11']


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
#Save Clarity data when using start and end from beginning to 7/31 (sensors on Paccar Roof)

Audubon_paccar_roof = solar_data['Audubon']    
Balboa_paccar_roof = solar_data['Balboa']  
Browne_paccar_roof = solar_data['Browne']  
Lidgerwood_paccar_roof = solar_data['Lidgerwood']  
Regal_paccar_roof = solar_data['Regal']  

Lidgerwood_1 = solar_data['Lidgerwood']

new = [-117.1548262,46.7295804]

for value in Lidgerwood_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    


with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Paccar_roof_until_7_31/Audubon_paccar_roof.json','w') as file:
    json.dump(Audubon_paccar_roof,file)
 
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Paccar_roof_until_7_31/Balboa_paccar_roof.json','w') as file:
    json.dump(Balboa_paccar_roof,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Paccar_roof_until_7_31/Browne_paccar_roof.json','w') as file:
    json.dump(Browne_paccar_roof,file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Paccar_roof_until_7_31/Lidgerwood_paccar_roof.json','w') as file:
    json.dump(Lidgerwood_paccar_roof,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Paccar_roof_until_7_31/Regal_paccar_roof.json','w') as file:
    json.dump(Regal_paccar_roof,file)



#%%
#Fixing Lat and Lon data
#when using start and end from 8/1 to 8/22 (sensors at facilities building in Spokane)
Audubon_1 = solar_data['Audubon']
Balboa_1 = solar_data['Balboa']
Browne_1 = solar_data['Browne']
Lidgerwood_1 = solar_data['Lidgerwood']
Regal_1 = solar_data['Regal']

#Spokane School District Facilities location
new = [-117.370646,47.693937]

for value in Audubon_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Balboa_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Browne_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Lidgerwood_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    
for value in Regal_1:
    dictionary = value
    dictionary['location']['coordinates'] = new
    

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Facilities_until_8_22/Audubon_Spokane_Facilities.json','w') as file:
    json.dump(Audubon_1,file)
 
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Facilities_until_8_22/Balboa_Spokane_Facilities.json','w') as file:
    json.dump(Balboa_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Facilities_until_8_22/Browne_Spokane_Facilities.json','w') as file:
    json.dump(Browne_1,file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Facilities_until_8_22/Lidgerwood_Spokane_Facilities.json','w') as file:
    json.dump(Lidgerwood_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Facilities_until_8_22/Regal_Spokane_Facilities.json','w') as file:
    json.dump(Regal_1,file)

#%%
#Fixing lat and lon data
#when using start and end from 8/22 to present (sensors at school locations)
Audubon_1 = solar_data['Audubon']
Balboa_1 = solar_data['Balboa']
Browne_1 = solar_data['Browne']
Lidgerwood_1 = solar_data['Lidgerwood']
Regal_1 = solar_data['Regal']

#School Latitudes and Longitudes

new_Audubon = [-117.441739,47.6798472]
new_Balboa = [-117.4560056,47.7181806]
new_Browne = [-117.4640639,47.70415]
new_Lidgerwood = [-117.405161,47.7081417]
new_Regal = [-117.369972,47.69735]

for value in Audubon_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Audubon
    
for value in Balboa_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Balboa
    
for value in Browne_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Browne
    
for value in Lidgerwood_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Lidgerwood
    
for value in Regal_1:
    dictionary = value
    dictionary['location']['coordinates'] = new_Regal
    

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Installed_at_School/Audubon_School.json','w') as file:
    json.dump(Audubon_1,file)
 
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Installed_at_School/Balboa_School.json','w') as file:
    json.dump(Balboa_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Installed_at_School/Browne_School.json','w') as file:
    json.dump(Browne_1,file)

with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Installed_at_School/Lidgerwood_School.json','w') as file:
    json.dump(Lidgerwood_1,file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Installed_at_School/Regal_Spokane_School.json','w') as file:
    json.dump(Regal_1,file)


#%%
#Combine 3 date dependent Locations into one JSON file for each "Batch 1" sensor   

import itertools

#test = pd.DataFrame({})
#files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Audubon*.json')
#files.sort()
#for file in files:
#    test = pd.concat([test, pd.read_json(file)], sort=False)
#test.index = test.time
#adams.index = adams.Datetime   

result = []

files = sorted(glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Audubon*.json"))
#f = open("bigfile.json", "w")
#for tempfile in files:
#    f.write(json.loads(tempfile))
for file in files:
    with open(file, "r") as infile:
        result.append(json.load(infile))
Audubon_backlog = list(itertools.chain.from_iterable(result))
Audubon_backlog = sorted(Audubon_backlog, key = lambda i: i['time'])

#%%
import itertools

result = []    
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Audubon*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Audubon_backlog = list(itertools.chain.from_iterable(result))
Audubon_backlog = sorted(Audubon_backlog, key = lambda i: i['time'])
#Audubon_backlog = json.dumps(Audubon_backlog)
#print(Audubon_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Audubon_Combined.json", "w") as outfile:
     json.dump(Audubon_backlog, outfile)
     
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Audubon_Combined.json') as json_file:
    Audubon_Combined=json.load(json_file)

result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Balboa*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Balboa_backlog = list(itertools.chain.from_iterable(result))
Balboa_backlog = sorted(Balboa_backlog, key = lambda i: i['time'])
#Balboa_backlog = json.dumps(Balboa_backlog)
#print(Balboa_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Balboa_Combined.json", "w") as outfile:
     json.dump(Balboa_backlog, outfile)
     
#with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Balboa_Combined.json') as json_file:
#    Balboa_Combined = json.load(json_file)
     
result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Browne*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Browne_backlog = list(itertools.chain.from_iterable(result))
Browne_backlog = sorted(Browne_backlog, key = lambda i: i['time'])
#Browne_backlog = json.dumps(Browne_backlog)
#print(Browne_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Browne_Combined.json", "w") as outfile:
     json.dump(Browne_backlog, outfile)
     
#with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Browne_Combined.json') as json_file:
#    Browne_Combined = json.load(json_file)

result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Lidgerwood*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Lidgerwood_backlog = list(itertools.chain.from_iterable(result))
Lidgerwood_backlog = sorted(Lidgerwood_backlog, key = lambda i: i['time'])
#Lidgerwood_backlog = json.dumps(Lidgerwood_backlog)
#print(Lidgerwood_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Lidgerwood_Combined.json", "w") as outfile:
     json.dump(Lidgerwood_backlog, outfile)
     
#with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Lidgerwood_Combined.json') as json_file:
#    Lidgerwood_Combined = json.load(json_file)
     
result = []
for f in glob("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Regal*.json"):
    with open(f, "r") as infile:
        result.append(json.load(infile))
Regal_backlog = list(itertools.chain.from_iterable(result))
Regal_backlog = sorted(Regal_backlog, key = lambda i: i['time'])
#Regal_backlog = json.dumps(Regal_backlog)
#print(Regal_backlog)
with open("/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Regal_Combined.json", "w") as outfile:
     json.dump(Regal_backlog, outfile)
     
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Regal_Combined.json') as json_file:
    Regal_Combined = json.load(json_file)
    
#%%
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Audubon_Combined.json') as json_file:
    Audubon_Combined=json.load(json_file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Balboa_Combined.json') as json_file:
    Balboa_Combined = json.load(json_file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Browne_Combined.json') as json_file:
    Browne_Combined = json.load(json_file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Lidgerwood_Combined.json') as json_file:
    Lidgerwood_Combined = json.load(json_file)
    
with open('/Users/matthew/Desktop/data/urbanova/ramboll/Clarity/Batch_1/Backlogged_Data_Combined/Regal_Combined.json') as json_file:
    Regal_Combined = json.load(json_file)