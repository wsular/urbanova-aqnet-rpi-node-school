#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:47:45 2020

@author: matthew
"""
import pandas as pd
import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
from glob import glob


# Choose dates of interest
start_time = '2019-12-17 15:00'
end_time = '2020-01-27 23:00'

interval = '15T'

Audubon_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Audubon*.csv')
files.sort()
for file in files:
    Audubon_All = pd.concat([Audubon_All, pd.read_csv(file)], sort=False)

Adams_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Adams*.csv')
files.sort()
for file in files:
    Adams_All = pd.concat([Adams_All, pd.read_csv(file)], sort=False)
    
Balboa_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Balboa*.csv')
files.sort()
for file in files:
    Balboa_All = pd.concat([Balboa_All, pd.read_csv(file)], sort=False)
    
Browne_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Browne*.csv')
files.sort()
for file in files:
    Browne_All = pd.concat([Browne_All, pd.read_csv(file)], sort=False)
    
Grant_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Grant*.csv')
files.sort()
for file in files:
    Grant_All = pd.concat([Grant_All, pd.read_csv(file)], sort=False)
    
Jefferson_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Jefferson*.csv')
files.sort()
for file in files:
    Jefferson_All = pd.concat([Jefferson_All, pd.read_csv(file)], sort=False)
    
Lidgerwood_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Lidgerwood*.csv')
files.sort()
for file in files:
    Lidgerwood_All = pd.concat([Lidgerwood_All, pd.read_csv(file)], sort=False)
    
Regal_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Regal*.csv')
files.sort()
for file in files:
    Regal_All = pd.concat([Regal_All, pd.read_csv(file)], sort=False)
    
Sheridan_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Sheridan*.csv')
files.sort()
for file in files:
    Sheridan_All = pd.concat([Sheridan_All, pd.read_csv(file)], sort=False)
    
Stevens_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Stevens*.csv')
files.sort()
for file in files:
    Stevens_All = pd.concat([Stevens_All, pd.read_csv(file)], sort=False)
    
Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv')
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)
    
Paccar_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Paccar*.csv')
files.sort()
for file in files:
    Paccar_All = pd.concat([Paccar_All, pd.read_csv(file)], sort=False)
    
Augusta_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)

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

Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]

Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar = Paccar_All.loc[start_time:end_time]

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time] 
    
#%%
import csv
import numpy as np

a = 1
b = 5
c = 9

e = 3
f = 5
g = 1


#d = [a,b,c]
#t = [e,f,g]
h = []
d = [1,2,3]
t = [4,5,6]



h.append(d)
h.append(t)

#npa = np.asarray(h)


with open('/Users/matthew/Desktop/test.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames = ['a','b','c'])
    writer.writeheader()
    f.close()

with open('/Users/matthew/Desktop/test.csv', 'a') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerows(h)
    
#with open ('/Users/matthew/Desktop/test.csv', 'a') as f:
#    np.savetxt(f, npa, delimiter=',')
#    f.close()
    
