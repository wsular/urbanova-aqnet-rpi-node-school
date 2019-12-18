#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:58:41 2019

@author: matthew
"""
import pandas as pd
from glob import glob
#%%

#Import entire data set

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
    
#%%
# choose schools to look at (comment out from full list to select schools of interest)
    #make sure that selection and selection names match
    
school_names = ['Adams', 
                'Jefferson', 
                'Grant',
                'Sheridan', 
                'Reference',
                'Stevens',
                'Regal', 
                'Audubon', 
                'Lidgerwood', 
                'Browne',
                'Balboa',
                'Paccar']    

selection = {'Adams':Adams_All,
             'Jefferson':Jefferson_All,
             'Grant':Grant_All,
             'Sheridan':Sheridan_All,
             'Reference':Reference_All,
             'Stevens':Stevens_All,
             'Regal':Regal_All,
             'Audubon':Audubon_All,
             'Lidgerwood':Lidgerwood_All,
             'Browne':Browne_All,
             'Balboa':Balboa_All,
             'Paccar':Paccar_All
              }

selection_names = ['Adams',
                   'Jefferson',
                   'Grant',
                   'Sheridan',
                   'Reference',
                   'Stevens',
                   'Regal',
                   'Audubon', 
                   'Lidgerwood',
                   'Browne',
                   'Balboa',
                   'Paccar'
                   ]
#%%
# Choose dates of interest
start_time = '2019-12-01 00:00'
end_time = '2019-12-02 00:00'

#%%
#create dataframes for selected sensors over desired time range

#resample to desired time interval so time series match
interval = '15T'


#%%
filtered_data = {}

for name in selection_names:
    filtered_data[name] = pd.DataFrame()

for i in filtered_data:
    School = selection[i]
    if School.equals(selection['Reference']):
        #down sampling because reference node is wired and has sampling frequencey ~ 2 min
        School['time'] = pd.to_datetime(School['time'])
        School = School.sort_values('time')                  #switches time so earliest date at beginning
        School.index = School.time                               #(so that can select by start, end not end, start times in that order)
        School = School.loc[start_time:end_time]
        School = School.resample(interval).mean()
    else:   
        
#        School = School.resample(interval).mean().interpolate(method='linear') #original method
    
    ### upsampling
    
        School['time'] = pd.to_datetime(School['time'])
        School = School.sort_values('time')                  #switches time so earliest date at beginning
        School.index = School.time                               #(so that can select by start, end not end, start times in that order)
        School = School.loc[start_time:end_time]
        School = School.resample(interval).pad()#().interpolate(method='linear')
        
    filtered_data[i] = School
#%%    

info = [[47.621172,  -117.367725,24.47,'Adams'],
        [47.621533,  -117.4098417,47.23,'Jefferson'],
        [47.6467083,  -117.390983,60.48,'Grant'],
        [47.6522472, -117.355561,79.46,'Sheridan'],
        [47.6608,  -117.4045056,37,'Reference'],
        [47.671256,  -117.3846583,42.3,'Stevens'],
        [47.69735,  -117.369972,26.94,'Regal'],
        [47.6798472,  -117.441739,18.52,'Audubon'],
        [47.7081417,  -117.405161,24.55,'Lidgerwood'],
        [47.70415, -117.4640639,31.66,'Browne'],
        [47.71818056,  -117.4560056,47.13,'Balboa']]

df = pd.DataFrame(info, columns =['Lat', 'Lon', 'PM2_5','Location'], dtype = float)

latlngbox = "47.612,-117.472,47.728,-117.348"   

#%%
import holoviews as hv
from holoviews import opts
hv.extension('bokeh', logo=False)
from bokeh.plotting import show
import geoviews as gv
import geoviews.tile_sources as gvts
gv.extension('bokeh', 'matplotlib', logo=False)


    
scatter = hv.Scatter(df.dropna(), kdims='Lon', vdims=['Lat', 'PM2_5', 'Location'])
scatter.opts(color='PM2_5', size=10, padding=.1, tools=['hover'], 
             colorbar=True, cmap='magma', width=500, height=400)#,clim=(0, 60))

show(hv.render(scatter))

#%%
points = gv.Points(df.dropna(), ['Lon', 'Lat'], ['PM2_5', 'Location'])

points.opts(size=10, color='PM2_5', cmap='magma', tools=['hover'], colorbar=True, 
            width=500, height=400, padding=.1)#, clim=(0, 60))

#labels = hv.Labels(df.dropna(), kdims=['Lon', 'Lat'], vdims=['Location'])

#labels = hv.Labels({('Lon', 'Lat'): df, 'text': df['Location']}, ['Lon', 'Lat'], 'text')

#(points* labels).opts(
#    opts.Labels(color='text', cmap='Category20', xoffset=0.05, yoffset=0.05, size=14, padding=0.2),
#    opts.Points(color='black', s=25))


#overlay = (points * labels)#.redim.range(x=(47.612,47.728), y=(-117.348,-117.472))

#test = gvts.OSM * overlay

test = gvts.OSM * points

#show(hv.render(overlay))
show(hv.render(test))
#%%    

import numpy as np

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from 
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi
    
#%%
    
latlngbox_num = list(map(float, latlngbox.split(',')))

lats = np.linspace(latlngbox_num[0], latlngbox_num[2], num=150)
lons = np.linspace(latlngbox_num[1], latlngbox_num[3], num=151)
meshgrid_shape = (lats.size, lons.size)

xi, yi = np.meshgrid(lons, lats)
xi, yi = xi.ravel(), yi.ravel()

df = df.dropna()
x, y = df.Lon.values, df.Lat.values
z = df.PM2_5.values

zi = simple_idw(x, y, z, xi, yi)
zi = zi.reshape(meshgrid_shape)    
    
interpolated = hv.Image(zi[::-1, :])
interpolated.opts(colorbar=True, alpha=0.7,cmap='magma', tools=['hover'], 
                   width=500, height=400)

test2 = (test + interpolated).cols(2)    
    
#show(hv.render(interpolated))  
show(hv.render(test2)) 

#%%



#show(hv.render(contour))

import xarray as xr

ds = xr.DataArray(zi, dims=['Lat', 'Lon'], 
                  coords={'Lat': lats, 'Lon': lons}).to_dataset(name='PM2_5')

aqi_ds = gv.Dataset(ds, ['Lon', 'Lat'], 'PM2_5')

background = gvts.OSM * gv.Image(aqi_ds).opts(alpha=0.7, width=500, height=400, 
                   colorbar=True, cmap='magma')
contour = gvts.CartoEco * aqi_ds.to(gv.FilledContours, 
        ['Lon', 'Lat']).opts(alpha=0.5, width=500, height=400, 
        colorbar=True, cmap='magma', levels=10, color_levels=10,
        tools=['hover'])

test3 = (test + background + contour).cols(2)
hv.save(test3, '/Users/matthew/Desktop/IDW.png', fmt='png')
show(hv.render(test3))
#show(hv.render(background))

#%%

np.random.seed(9)
data = np.random.rand(10, 2)
points = hv.Points(data)
labels = hv.Labels({('x', 'y'): data, 'text': [chr(65+i) for i in range(10)]}, ['x', 'y'], 'text')
overlay = (points * labels).redim.range(x=(-0.2, 1.2), y=(-.2, 1.2))

overlay.opts(
    opts.Labels(text_font_size='10pt', xoffset=0.08),
    opts.Points(color='black', size=5))

test4 = (overlay + points).cols(2)

show(hv.render(test4))










