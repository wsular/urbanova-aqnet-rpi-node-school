#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:58:41 2019

@author: matthew
"""
import pandas as pd
from glob import glob
import numpy as np
import xarray as xr
import holoviews as hv
from holoviews import opts


from bokeh.plotting import show
import geoviews as gv
import geoviews.tile_sources as gvts


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

hv.extension('bokeh', logo=False)
gv.extension('bokeh', logo=False)

#hv.extension('matplotlib', logo=False)
#gv.extension('matplotlib', logo=False)
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
                'Paccar'
             ]    

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

selection_names = ['Adams',           # 47.621172,  -117.367725,24.47
                   'Jefferson',       # 47.621533,  -117.4098417,47.23
                   'Grant',           # 47.6467083,  -117.390983,60.48
                   'Sheridan',        # 47.6522472, -117.355561
                   'Reference',       # 47.6608,  -117.4045056
                   'Stevens',         # 47.671256,  -117.3846583
                   'Regal',           # 47.69735,  -117.369972
                   'Audubon',         # 47.6798472,  -117.441739
                   'Lidgerwood',      # 47.7081417,  -117.405161
                   'Browne',          # 47.70415, -117.4640639
                   'Balboa' ,         # 47.70415, -117.4640639
                   'Paccar'           # 
                   ]
#%%
# Choose dates of interest
start_time = '2019-10-09 00:00'
end_time = '2019-10-10 00:00'

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
        #School = School.dropna()
        
    filtered_data[i] = School

Adams = filtered_data['Adams']
Adams['Lat']='47.621172'
Adams['Lon'] = '-117.367725'

Jefferson = filtered_data['Jefferson']
Jefferson['Lat']='47.621533'
Jefferson['Lon'] = '-117.4098417'

Grant = filtered_data['Grant']
Grant['Lat']='47.6467083'
Grant['Lon'] = '-117.390983'

Sheridan = filtered_data['Sheridan']
Sheridan['Lat']='47.6522472'
Sheridan['Lon'] = '-117.355561'

Reference = filtered_data['Reference']
Reference['Lat']='47.6608'
Reference['Lon'] = '-117.4045056'
Reference['Location'] = 'Reference'
Reference['time'] = Adams['time']
Reference['ID'] = 'AKYWSPV0'


Stevens = filtered_data['Stevens']
Stevens['Lat']='47.671256'
Stevens['Lon'] = '-117.3846583'

Regal = filtered_data['Regal']
Regal['Lat']='47.69735'
Regal['Lon'] = '-117.369972'

Audubon = filtered_data['Audubon']
Audubon['Lat']='47.6798472'
Audubon['Lon'] = '-117.441739'

Lidgerwood = filtered_data['Lidgerwood']
Lidgerwood['Lat']='47.7081417'
Lidgerwood['Lon'] = '-117.405161'

Browne = filtered_data['Browne']
Browne['Lat']='47.70415'
Browne['Lon'] = '-117.4640639'

Balboa = filtered_data['Balboa']
Balboa['Lat']='47.71818056'
Balboa['Lon'] = '-117.4560056'

#Paccar = filtered_data['Paccar']
#Paccar['Lat']='47.71818056'
#Paccar['Lon'] = '-117.4560056'

#
#%%
#info = []

data_frames = [Adams, Jefferson, Grant, Sheridan, Reference, Stevens, Regal, Audubon, Lidgerwood, Browne, Balboa]

new_df = pd.concat(data_frames)

new_df.sort_index(inplace=True)
new_df = new_df.dropna()
del new_df['ID']
del new_df['PM10']
del new_df['Rel_humid']
del new_df['temp']
del new_df['time']
#new_df['time'] = new_df.index
new_df = new_df[['Lat','Lon','PM2_5','Location']]
new_df['Lat'] = new_df['Lat'].astype(float)
new_df['Lon'] = new_df['Lon'].astype(float)
new_df['PM2_5'].astype(float)
print(new_df.dtypes)
#minute = new_df.index.to_period("m")
#agg = new_df.groupby([minute])
#for group in agg:
#    info = group
#    print(info)

#%%

# manually put in mercator northing and easting for school labels

easting = '-13070000'
northing = '6050000'


Adams1 = filtered_data['Adams']
Adams1['Lat']= '6044100'
Adams1['Lon'] = '-13062900'

Jefferson1 = filtered_data['Jefferson']
Jefferson1['Lat']= '6044200'
Jefferson1['Lon'] = '-13073000'

Grant1 = filtered_data['Grant']
Grant1['Lat']= '6048400'
Grant1['Lon'] = '-13070000'

Sheridan1 = filtered_data['Sheridan']
Sheridan1['Lat'] = '6048100'
Sheridan1['Lon'] = '-13062300'

Reference1 = filtered_data['Reference']
Reference1['Location'] = 'Reference'
Reference1['time'] = Adams1['time']
Reference1['ID'] = 'AKYWSPV0'
Reference1['Lat']= '6050600'
Reference1['Lon'] = '-13073000'

Stevens1 = filtered_data['Stevens']
Stevens1['Lat']= '6052300'
Stevens1['Lon'] = '-13064300'

Regal1 = filtered_data['Regal']
Regal1['Lat']= '6056700'
Regal1['Lon'] = '-13063400'

Audubon1 = filtered_data['Audubon']
Audubon1['Lat']= '6053850'
Audubon1['Lon'] = '-13076600'

Lidgerwood1 = filtered_data['Lidgerwood']
Lidgerwood1['Lat']= '6059950'
Lidgerwood1['Lon'] = '-13068000'

Browne1 = filtered_data['Browne']
Browne1['Lat']= '6057850'
Browne1['Lon'] = '-13078600'

Balboa1 = filtered_data['Balboa']
Balboa1['Lat']= '6061500'
Balboa1['Lon'] = '-13075300'


data_frames1 = [Adams1, Jefferson1, Grant1, Sheridan1, Reference1, Stevens1, Regal1, Audubon1, Lidgerwood1, Browne1, Balboa1]

new_df1 = pd.concat(data_frames1)
new_df1.sort_index(inplace=True)

new_df1 = new_df1.dropna()


del new_df1['ID']
del new_df1['PM10']
del new_df1['Rel_humid']
del new_df1['temp']
del new_df1['time']
#new_df['time'] = new_df.index
new_df1 = new_df1[['Lat','Lon','PM2_5','Location']]
new_df1['Lat'] = new_df1['Lat'].astype(float)
new_df1['Lon'] = new_df1['Lon'].astype(float)
new_df1['PM2_5'].astype(float)
#%%
#for group in agg:
#    if location.equals(filtered_data['Adams']):
#        info_row = [47.621172,  -117.367725,row[location]['PM2_5'],'Adams']
##        info.append(info_row)
 #       
 #   if location.equals(filtered_data['Jefferson']):
 #      print('3')
 #      nfo_row = [47.621533,  -117.4098417,row[location]['PM2_5'],'Jefferson']
 #      info.append(info_row)
 #   print(group)

#%%

#For Bokeh interactive

from holoviews.plotting.bokeh.styles import (line_properties, fill_properties, text_properties)
import datetime
#from bokeh.models import ColorBar, LogColorMapper
#from bokeh.io import export_png
#from bokeh.io import export_svgs
#from bokeh.io.export import get_screenshot_as_png
#import matplotlib
#hv.extension('matplotlib')
#hv.Store.renderers
#from bokeh.models import Label
#from bokeh.plotting import figure
#from bokeh.models import Panel, Tabs
#from bokeh.io import output_notebook, output_file, show


begin = '2019-10-09 00:15'
#begin = start_time    
begin = datetime.datetime.strptime(begin, '%Y-%m-%d %H:%M')

end = end_time
end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M')

t = datetime.timedelta(minutes=15)

while begin < end:
     
    #mask = (new_df['time'] == begin)
    #info = new_df.loc[mask]
    
    df = new_df.loc[begin]
    df1 = new_df1.loc[begin]
    
    latlngbox = "47.612,-117.472,47.728,-117.348" 
    
    points = gv.Points(df.dropna(), ['Lon', 'Lat'], ['PM2_5', 'Location'])
    print((points))
    
    levels = 10#[0,10,20,30,40,50,60]
    labels = hv.Labels(df1, kdims = ['Lon', 'Lat'], vdims =['Location'])
    labels.opts(text_color = 'white')
    #levels = np.arange(0,100,6)
    points.opts(size=10, color='PM2_5', cmap='magma', clim=(0,70),
                color_levels = levels,
                colorbar_opts={
        'major_label_overrides': {
            0:'0',
            10:'10',
            20:'20',
            30:'30',
            35:'ug/m^3',
            40:'40',
            50:'50',
            60:'60',
            70:'70',
        },
        'major_label_text_align': 'left',
    },
                tools=['hover'], colorbar=True, 
            width=500, height=400, padding=.1, title = begin.strftime("%m/%d/%Y %H:%M"))
    
    
    test = gvts.EsriImagery * points * labels #* gv.Labels(points, kdims=['Lat','Lon'], vdims =['Location'])
   
    #test = test * hv.Text(47.715,-117.35, begin.strftime("%m/%d/%Y, %H:%M"))
    
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
    
    #interpolated = hv.Image(zi[::-1, :])
    #interpolated.opts(colorbar=True, alpha=0.7,cmap='magma', tools=['hover'], 
    #               width=500, height=400)

    ds = xr.DataArray(zi, dims=['Lat', 'Lon'], 
                  coords={'Lat': lats, 'Lon': lons}).to_dataset(name='PM2_5')

    aqi_ds = gv.Dataset(ds, ['Lon', 'Lat'], 'PM2_5')

    background = gvts.EsriImagery * gv.Image(aqi_ds).opts(alpha=0.7, width=500, height=400, 
                   colorbar=True, cmap='magma',clim=(0,70), color_levels = levels)
    
    contour = gvts.EsriReference * aqi_ds.to(gv.FilledContours,         #gvts.CartoEco is the original
        ['Lon', 'Lat']).opts(alpha=0.5, width=500, height=400, 
        colorbar=True, cmap='magma', levels=10, color_levels=10,
        tools=['hover'])
    
    text_opts  = opts.Text(text_align='center', text_baseline='middle', text_font='Arial',width = 500, height = 400)
                             
    test3 = (test + background + contour + 
             hv.Text(0.5, 0.5, begin.strftime("%m/%d/%Y %H:%M")).opts(text_opts)).cols(2)
    print(type(test))
    print(type(background))
    print(type(contour))
    print(type(test3))
    
    #renderer = hv.Store.renderers['matplotlib'].instance(fig='svg', holomap='gif')
    #renderer.save(test3, 'test3')
    
    #renderer = hv.plotting.mpl.MPLRenderer.instance(dpi=120)
    #layout_test3 = renderer.get_plot(test3)
    #print(type(layout_test3))
    #print(type(test3))
    # Using bokeh
   # hv.render(test3)
    hv.save(test3, '/Users/matthew/Desktop/IDW/html/10_9_to_10_19_19/IDW_test' + begin.strftime('%Y_%m_%d_%H_%M') + '.html', backend='bokeh')    # works
    
    #image = get_screenshot_as_png(layout_test3, backend = 'bokeh')#, height=height, width=width, driver=webdriver)
    
    #hv.save(test3, '/Users/matthew/Desktop/IDW/IDW_test' + begin.strftime('%Y-%m-%d %H:%M') + '.png')
    #hv.save(test3, '/Users/matthew/Desktop/IDW/DIDW_test' + begin.strftime('%Y-%m-%d %H:%M') + '.svg', backend='matplotlib')
    #test3.output_backend = "svg"
   # export_svgs(test3, filename='/Users/matthew/Desktop/IDW/DIDW_test' + begin.strftime('%Y-%m-%d %H:%M') + '.svg)
    
    #export_png(test3, filename='/Users/matthew/Desktop/IDW/IDW_test' + begin.strftime('%Y_%m_%d_%H_%M') + '.png')
    hv.save(test3.options(toolbar=None), '/Users/matthew/Desktop/IDW/png/10_9_to_10_17_19/png' + begin.strftime('%Y_%m_%d_%H_%M') + '.png', fmt='png', backend='bokeh')    # works
    #show(hv.render(test3))
    
    print(begin)
    
    begin =  begin + t
#    for row in new_df['time']:
#        
#        check = new_df.loc[row]['time']
#        print(check)
        #check = datetime.datetime.strptime(check,'%Y-%m-%d %H:%M:%S')
        #print(type(check))
       # print(type(begin))
#        print('1')
        #if check == begin:
        #    print('2')
  #          row = [new_df['lat'],new_df['lon'],new_df['PM2_5'],new_df['location']]
  #          info.append(row)
#%%            
    

#            begin = begin + 
#            start_time = start_time
#            create intervaled dataframe here
#            append dataframe to dataframes list
#            start_time + interval

        
#new loop
#for df in dataframes:
#    info = df
#    plotting code here
    
#%%    

# test building dataframe by iterating through dictionary of data frames line by line to generate image of each time interval
# NOT USED

info = []

for location in filtered_data:
    location = filtered_data[location]
    print('1')
   # for row in location:
    if location.equals(filtered_data['Adams']):
       print('2')
       location['Lat']='47.621172'
       location['Lon'] = '-117.367725'
       
       #for row in location:
#       info_row = [47.621172,  -117.367725,row[location]['PM2_5'],'Adams']
#       info.append(info_row)
    if location.equals(filtered_data['Jefferson']):
       print('3')
       location['Lat']='47.621533'
       location['Lon'] = '117.4098417'
       #for row in location:
 #      info_row = [47.621533,  -117.4098417,row[location]['PM2_5'],'Jefferson']
 #      info.append(info_row)
    if location.equals(filtered_data['Grant']):
       print('4')
       location['Lat']='47.6467083'
       location['Lon'] = '-117.390983'
       #for row in location:
  #     info_row = [47.6467083,  -117.390983, row[location]['PM2_5'],'Grant']
  #     info.append(info_row)
#%%
# Info for testing IDW       
       
       
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
print(df.dtypes)
latlngbox = "47.612,-117.472,47.728,-117.348"   

#%%

points = gv.Points(df.dropna(), ['Lon', 'Lat'], ['PM2_5', 'Location'])

points.opts(size=10, color='PM2_5', cmap='magma', tools=['hover'], colorbar=True, 
            width=500, height=400, padding=.1)
    
test = gvts.OSM * points
    
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
    #print(type(test3))
    # Using bokeh

    #export_png(test3, filename='/Users/matthew/Desktop/IDW/IDW_test' + begin.strftime('%Y_%m_%d_%H_%M') + '.png')
    #hv.save(test3, '/Users/matthew/Desktop/IDW' + begin.strftime('%Y-%m-%d %H:%M') + '.png', fmt='png')
show(hv.render(test3))
#%%



# Test plotting a scatter  plot of sensor locations and color scaled PM 2.5 concentrations
scatter = hv.Scatter(df.dropna(), kdims='Lon', vdims=['Lat', 'PM2_5', 'Location'])
scatter.opts(color='PM2_5', size=10, padding=.1, tools=['hover'], 
             colorbar=True, cmap='magma', width=500, height=400)#,clim=(0, 60))
print(type(scatter))

hv.save(scatter, '/Users/matthew/Desktop/IDW/IDW_test' + begin.strftime('%Y-%m-%d %H:%M') + '.png')
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

#map layer with locations on top of a background map
test = gvts.OSM * points
#print(type(test))
#show(hv.render(overlay))
#show(hv.render(test))

    
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

#test2 = (test + interpolated).cols(2)    
    
#show(hv.render(interpolated))  
#show(hv.render(test2)) 

#show(hv.render(contour))

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


#hv.renderer('bokeh').save(test3, '/Users/matthew/Desktop/IDW_test.png', fmt='png')

hv.save(hv.render(test3), '/Users/matthew/Desktop/IDW_test.png', fmt='png')

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

#show(hv.render(test4))


#%%
import imgkit

imgkit.from_file('/Users/matthew/Desktop/IDW/DIDW_test2019_10_01_23_45.html', '/Users/matthew/Desktop/IDW/test.jpg')
#%%
import pdfkit
#%%
path_wkhtmltopdf = r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe'
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
pdfkit.from_url("http://google.com", "out.pdf", configuration=config)


#%%
import os
os.environ['PYTHONPATH']
'/home/my_user/code'

#%%
import weasyprint

weasyprint('/Users/matthew/Desktop/IDW/DIDW_test2019_10_01_23_45.html', '/Users/matthew/Desktop/IDW/test.jpg')

#%%

import subprocess
#subprocess.call(['python',"webkit2png.py", '/Users/matthew/Desktop/IDW/DIDW_test2019_10_01_23_45.html'])


def scrape_url(url, outpath):
    """
    Requires webkit2png to be on the path
    """
    subprocess.call(["webkit2png", "-o", outpath, "-g", "1000", "1260",
                     "-t", "30", url])

scrape_url('/Users/matthew/Desktop/IDW/DIDW_test2019_10_01_23_45.html','/Users/matthew/Desktop/IDW/test.jpg')
