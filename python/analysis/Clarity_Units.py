#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 15:17:51 2019

@author: matthew
"""

###### SOMETIMES THE CHOSEN DATE RANGE CAUSES AN ERROR "cannot reindex a non-unique index with a method or limit"

#%%
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
#Read in SRCAA Augusta site BAM data

Augusta_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)
    
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
start_time = '2020-02-08 00:00'
end_time = '2020-02-11 11:00'
#%%
import pandas as pd
from glob import glob
# Plot Grant to check how often data frequency is dropping

grant = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Grant*.csv')
files.sort()
for file in files:
    grant = pd.concat([grant, pd.read_csv(file)], sort=False)
    
grant['time'] = pd.to_datetime(grant['time'])
grant = grant.sort_values('time')
grant.index = grant.time
grant = grant.loc[start_time:end_time]



from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure

PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/Grant_data_frequency_check.html')
    
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(grant.index,     grant.PM2_5,  legend='Grant',       color='green',     line_width=2)

tab1 = Panel(child=p1, title="PM 2.5")

tabs = Tabs(tabs=[ tab1])

show(tabs)


#%%
#Jefferson Comparison Data for indoor PMS5003 unit and Clarity Unit overlap
interval = '2T'


jefferson = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Jefferson/WSU_LAR_Indoor_Air_Quality_Node_8_202002*.csv')
files.sort()
for file in files:
    jefferson = pd.concat([jefferson, pd.read_csv(file)], sort=False)
jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson = jefferson.sort_values('Datetime')
jefferson.index = jefferson.Datetime
jefferson = jefferson.resample(interval).mean()

Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar = Paccar_All.loc[start_time:end_time]

Jefferson_All['time'] = pd.to_datetime(Jefferson_All['time'])
Jefferson_All = Jefferson_All.sort_values('time')
Jefferson_All.index = Jefferson_All.time
Jefferson = Jefferson_All.loc[start_time:end_time]

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure

PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/Jefferson_indoor_comparison.html')
    
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(jefferson.index,     jefferson.PM2_5_standard,  legend='PMS5003',       color='green',     line_width=2)
#p1.line(Paccar.index,        Paccar.PM2_5,              legend='Clarity',       color='blue',      line_width=2) 
p1.line(Jefferson.index,     Jefferson.PM2_5,           legend='Outside',       color='red',       line_width=2) 


tab1 = Panel(child=p1, title="PM 2.5")

tabs = Tabs(tabs=[ tab1])

show(tabs)


#%%
jefferson.to_csv('/Users/matthew/Desktop/PMS5003_data.csv')
Paccar.to_csv('/Users/matthew/Desktop/Clarity_unit_indoors_data.csv')
Jefferson.to_csv('/Users/matthew/Desktop/Clarity_outdoor_data.csv')
#%%

#plot Browne Unit on paccar roof to determine its status
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure


Browne_All['time'] = pd.to_datetime(Browne_All['time'])
Browne_All = Browne_All.sort_values('time')
Browne_All.index = Browne_All.time
Browne = Browne_All.loc[start_time:end_time]


PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/Browne_Paccar_Roof.html')
    
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(Browne.index,     Browne.PM2_5,  legend='Browne',       color='green',     line_width=2)

tab1 = Panel(child=p1, title="Browne PM 2.5 Paccar Roof")

tabs = Tabs(tabs=[ tab1])

show(tabs)




#%%
#create dataframes for selected sensors over desired time range

#resample to desired time interval so time series match
interval = '15T'
#%%
#Check against original data
Audubon_All['time'] = pd.to_datetime(Audubon_All['time'])
Audubon_All = Audubon_All.sort_values('time') 
Audubon_All.index = Audubon_All.time
del(Audubon_All['time'])
Audubon_cut = Audubon_All.loc[start_time:end_time]

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
#combine into 1 csv per location to send to Von 11/16/19

#Reference_site.to_csv('/Users/matthew/Desktop/data/Clarity_Backup/Reference_site' + '_' + date_range + '.csv', index=False)
#Paccar.to_csv('/Users/matthew/Desktop/data/Clarity_Backup/Paccar' + '_' + date_range + '.csv', index=False)
Audubon.to_csv('/Users/matthew/Desktop/Audubon.csv', index=False)
Adams.to_csv('/Users/matthew/Desktop/Adams.csv', index=False)
Balboa.to_csv('/Users/matthew/Desktop/Balboa.csv', index=False)
Browne.to_csv('/Users/matthew/Desktop/Browne.csv', index=False)
Grant.to_csv('/Users/matthew/Desktop/Grant.csv', index=False)
Jefferson.to_csv('/Users/matthew/Desktop/Jefferson.csv', index=False)
Lidgerwood.to_csv('/Users/matthew/Desktop/Liderwood.csv', index=False)
Regal.to_csv('/Users/matthew/Desktop/Regal.csv', index=False)
Sheridan.to_csv('/Users/matthew/Desktop/Sheridan.csv', index=False)
Stevens.to_csv('/Users/matthew/Desktop/Stevens.csv', index=False)
Reference.to_csv('/Users/matthew/Desktop/Reference_Site.csv', index=False)
#Paccar.to_csv('/Users/matthew/Desktop/data/Clarity_Backup/Paccar' + '_' + date_range + '.csv', index=False)    
#%%
# Make individual Datafarmes for analysis    (resample to 15 min intervals so consistent data times can be compared)
# And then combine parameter of interest into combined data frame for

#Choose Parameters

parameters = []

parameter1 = 'PM2_5'
parameter2 = 'PM10'
parameter3 = 'temp'
parameter4 = 'Rel_humid' 

#%%
if parameter1 in locals() or globals():
    parameters.append(parameter1)
    Combined_data_PM2_5 = pd.DataFrame()
  
if parameter2 in locals() or globals():
    parameters.append(parameter2)
    Combined_data_PM10 = pd.DataFrame()
    
if parameter3 in locals() or globals():
    parameters.append(parameter3)
    Combined_data_temp = pd.DataFrame()
    
if parameter4 in locals() or globals():
    parameters.append(parameter4)
    Combined_data_Rel_humid = pd.DataFrame()
#%%
if parameter1 in parameters:

    for i in filtered_data:
            Combined_data_PM2_5[i] = filtered_data[i][parameter1]
#%%
if parameter1 in parameters:

    for i in filtered_data:
        if 'Audubon' in list(filtered_data.keys()):
            Combined_data_PM2_5['Audubon'] = filtered_data['Audubon'][parameter1]   
       
        if 'Adams' in list(filtered_data.keys()):
            Combined_data_PM2_5['Adams'] = filtered_data['Adams'][parameter1]    
    
        if 'Balboa' in list(filtered_data.keys()):
            Combined_data_PM2_5['Balboa'] = filtered_data['Balboa'][parameter1]    
        
        if 'Browne' in list(filtered_data.keys()):
            Combined_data_PM2_5['Browne'] = filtered_data['Browne'][parameter1]    
        
        if 'Grant' in list(filtered_data.keys()):
            Combined_data_PM2_5['Grant'] = filtered_data['Grant'][parameter1]    
        
        if 'Jefferson' in list(filtered_data.keys()):
            Combined_data_PM2_5['Jefferson'] = filtered_data['Jefferson'][parameter1]    
        
        if 'Lidgerwood' in list(filtered_data.keys()):
            Combined_data_PM2_5['Lidgerwood'] = filtered_data['Lidgerwood'][parameter1]  
        
        if 'Regal' in list(filtered_data.keys()):
            Combined_data_PM2_5['Regal'] = filtered_data['Regal'][parameter1]    
        
        if 'Sheridan' in list(filtered_data.keys()):
            Combined_data_PM2_5['Sheridan'] = filtered_data['Sheridan'][parameter1]   
        
        if 'Stevens' in list(filtered_data.keys()):
            Combined_data_PM2_5['Stevens'] = filtered_data['Stevens'][parameter1]   
            
        if 'Reference' in list(filtered_data.keys()):
            Combined_data_PM2_5['Reference'] = filtered_data['Reference'][parameter1]   
#%%
# save csv of combined PM 2.5 data for check in excel if heatmap calculated correctly
Combined_data_PM2_5.to_csv('/Users/matthew/Desktop/11_8_to_12_3_19.csv')
#%%
#Compute mean PM 2.5 of each location and plot bar chart
from matplotlib import pyplot as plt
           
average_PM2_5_September = {'Audubon':Combined_data_PM2_5['Audubon'].mean(),
                           'Adams':Combined_data_PM2_5['Adams'].mean(),
                           'Balboa':Combined_data_PM2_5['Balboa'].mean(),
                           'Browne':Combined_data_PM2_5['Browne'].mean(),
                           'Grant':Combined_data_PM2_5['Grant'].mean(),
                           'Jefferson':Combined_data_PM2_5['Jefferson'].mean(),
                           'Lidgerwood':Combined_data_PM2_5['Lidgerwood'].mean(),
                           'Regal':Combined_data_PM2_5['Regal'].mean(),
                           'Sheridan':Combined_data_PM2_5['Sheridan'].mean(),
                           'Stevens':Combined_data_PM2_5['Stevens'].mean(),
                           'Reference':Combined_data_PM2_5['Reference'].mean()}

average_PM2_5_September = pd.DataFrame([average_PM2_5_September], columns=average_PM2_5_September.keys())

ax = average_PM2_5_September[['Audubon',
                              'Adams',
                              'Balboa',
                              'Browne',
                              'Grant',
                              'Jefferson',
                              'Lidgerwood',
                              'Regal',
                              'Sheridan',
                              'Stevens',
                              'Reference']].plot(kind='bar',figsize=(11.75, 8))#, 
                              #title ="Overall PM 2.5", figsize=(15, 10), legend=True, fontsize=20)
ax.set_xlabel('School', fontsize = 20)
ax.set_ylabel('PM 2.5 (ug/m^3)', fontsize = 20)
ax.set_title('10/9-10/17 Avg PM 2.5', fontsize = 28)
plt.tick_params(labelsize=20)
#SMALL_SIZE = 15
#MEDIUM_SIZE = 15
#BIGGER_SIZE = 24

#plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


plt.savefig('/Users/matthew/Desktop/Event_average_PM2_5.png')
#drop values below 30 
            
#%%

if parameter2 in parameters:

    for i in filtered_data:
        if 'Audubon' in list(filtered_data.keys()):
            Combined_data_PM10['Audubon'] = filtered_data['Audubon'][parameter2]   
       
        if 'Adams' in list(filtered_data.keys()):
            Combined_data_PM10['Adams'] = filtered_data['Adams'][parameter2]    
    
        if 'Balboa' in list(filtered_data.keys()):
            Combined_data_PM10['Balboa'] = filtered_data['Balboa'][parameter2]    
        
        if 'Browne' in list(filtered_data.keys()):
            Combined_data_PM10['Browne'] = filtered_data['Browne'][parameter2]    
        
        if 'Grant' in list(filtered_data.keys()):
            Combined_data_PM10['Grant'] = filtered_data['Grant'][parameter2]    
        
        if 'Jefferson' in list(filtered_data.keys()):
            Combined_data_PM10['Jefferson'] = filtered_data['Jefferson'][parameter2]    
        
        if 'Lidgerwood' in list(filtered_data.keys()):
            Combined_data_PM10['Lidgerwood'] = filtered_data['Lidgerwood'][parameter2]  
        
        if 'Regal' in list(filtered_data.keys()):
            Combined_data_PM10['Regal'] = filtered_data['Regal'][parameter2]    
        
        if 'Sheridan' in list(filtered_data.keys()):
            Combined_data_PM10['Sheridan'] = filtered_data['Sheridan'][parameter2]   
        
        if 'Stevens' in list(filtered_data.keys()):
            Combined_data_PM10['Stevens'] = filtered_data['Stevens'][parameter2]  
            
        if 'Reference' in list(filtered_data.keys()):
            Combined_data_PM10['Reference'] = filtered_data['Reference'][parameter2] 



#%%
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.transforms


def heat_map(data, **kwargs):
    col_map = kwargs.get('color_palette', sns.light_palette('orange', n_colors=7, as_cmap=False))
    plt.figure(figsize=(12,10))
    
    ax = sns.heatmap(
        vmin=0.65,
        vmax=1,
        data=data,
        cmap=col_map,
        cbar_kws={'ticks': [0.6,0.7, 0.8, 0.9,1.0]},   # for 4 colors and vmin = 0.6
        #cbar_kws={'ticks': [0.6,0.7, 0.8, 0.9,1.0]},   # for 4 colors and vmin = 0.6
        #xticklabels=labels,
        #yticklabels=labels,
        linewidths=0.75,
        annot=True, annot_kws={"size":16}
    ) 
    
    ax.set_title('11/8 to 12/03/19 PM 2.5', fontsize = 16)
    plt.xticks(range(Combined_data_PM2_5.shape[1]), Combined_data_PM2_5.columns, fontsize=14, rotation=45)
    plt.yticks(range(Combined_data_PM2_5.shape[1]), Combined_data_PM2_5.columns, fontsize=14)
    
    labels = Combined_data_PM2_5.columns

    data = np.random.random((10,10))

# Shift ticks to be at 0.5, 1.5, etc
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set(ticks=np.arange(0.5, len(labels)), ticklabels=labels)

    
#f = plt.figure(figsize=(19, 15))
#plt.matshow(Combined_data_PM2_5.corr(), fignum=f.number)
#plt.xticks(range(Combined_data_PM2_5.shape[1]), Combined_data_PM2_5.columns, fontsize=14, rotation=45)
#plt.yticks(range(Combined_data_PM2_5.shape[1]), Combined_data_PM2_5.columns, fontsize=14)
#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=14)
#plt.title('Correlation Matrix', fontsize=16);
#heat_map(corr)    
heat_map(Combined_data_PM2_5.corr())   

plt.savefig('/Users/matthew/Desktop/test.png')
#plt.savefig('/Users/matthew/Desktop/11_8_to_12_03_19_PM2_5_corr_matrix.png')
#%% 
corr = Combined_data_PM2_5.corr()
corr[np.abs(corr)>.99] = 0  
#%%
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

boundaries = [0,0.5, 0.6, 0.99,1]#0.7, 0.8, 0.9, 0.99, 1.0]  # custom boundaries

# here I generated twice as many colors, 
# so that I could prune the boundaries more clearly
hex_colors = sns.light_palette('red', n_colors=len(boundaries) * 2 + 2, as_cmap=False).as_hex()
hex_colors = [hex_colors[i] for i in range(0, len(hex_colors), 2)]

colors=list(zip(boundaries, hex_colors))

custom_color_map = LinearSegmentedColormap.from_list(
    name='custom_navy',
    colors=colors,
)

sns.heatmap(
        vmin=0.5,
        vmax=1.0,
        data=Combined_data_PM2_5.corr(),
        cmap=custom_color_map,
    #    xticklabels=labels,
   #     yticklabels=labels,
        linewidths=0.5,
        linecolor='lightgray'
  )            

# Manually specify colorbar labelling after it's been generated
#colorbar = ax.collections[0].colorbar
#colorbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1])
#colorbar.set_ticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '0.99', '1'])     
            
#%%
# Test to make discrete bins for heatmap

import matplotlib.pyplot as plt

import seaborn.apionly as sns
from matplotlib.colors import LinearSegmentedColormap

sns.set(font_scale=0.8)

# For only three colors, it's easier to choose them yourself.
# If you still really want to generate a colormap with cubehelix_palette instead,
# add a cbar_kws={"boundaries": linspace(-1, 1, 4)} to the heatmap invocation
# to have it generate a discrete colorbar instead of a continous one.
myColors = ((0.8, 0.8, 0.0, 1.0), (0.0, 0.8, 0.0, 1.0), (0.0, 0.0, 0.8, 1.0))
cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

ax = sns.heatmap(Combined_data_PM2_5.corr(), cmap=cmap, linewidths=.5, linecolor='lightgray')

# Manually specify colorbar labelling after it's been generated
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([-0.667, 0, 0.667])
colorbar.set_ticklabels(['B', 'A', 'C'])

# X - Y axis labels
#ax.set_ylabel('FROM')
#ax.set_xlabel('TO')

# Only y-axis labels need their rotation set, x-axis labels already have a rotation of 0
_, labels = plt.yticks()
plt.setp(labels, rotation=0)

plt.show()
            
#%%            
            
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,10))

ax = sns.heatmap(Combined_data_PM2_5.corr(),
            vmin=0.5,
            cmap='YlOrRd',
  #          n_colors = 5,
            annot=True, annot_kws={"size":16});
ax.set_title('September PM 2.5')
plt.savefig('/Users/matthew/Desktop/September_PM2_5_corr_matrix.png')
#%%
#Create Correlation Matrix for PM 2.5

corr_PM2_5 = Combined_data_PM2_5.corr()    
#corr_PM2_5.style.background_gradient(cmap='coolwarm', axis=None)

#%%            
            
            
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure

PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/October.html')
#Plot Clarity Batch 1 and EBAM overlap data


p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')

if 'Audubon' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Audubon,  legend='Audubon',       color='green',     line_width=2)
    #p1.line(Audubon_cut.index,                       Audubon_cut.PM2_5,  legend='15 min interp',    color='red',      line_width=2)

if 'Adams' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Adams,  legend='Adams',       color='red',     line_width=2)    

if 'Balboa' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Balboa,  legend='Balboa',       color='blue',     line_width=2)    

if 'Browne' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Browne,  legend='Browne',       color='brown',     line_width=2)  
    
if 'Grant' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Grant,  legend='Grant',       color='yellow',     line_width=2)   
    
if 'Jefferson' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Jefferson,  legend='Jefferson',       color='gold',     line_width=2)    

if 'Lidgerwood' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Lidgerwood,  legend='Lidgerwood',       color='magenta',     line_width=2)    

if 'Regal' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Regal,  legend='Regal',       color='lime',     line_width=2)    

if 'Sheridan' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Sheridan,  legend='Sheridan',       color='black',     line_width=2)    

if 'Stevens' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Stevens,  legend='Stevens',       color='orange',     line_width=2)    

if 'Reference' in list(filtered_data.keys()):            
    p1.line(Combined_data_PM2_5.index,              Combined_data_PM2_5.Reference,  legend='Reference',       color='purple',     line_width=2)    

    p1.legend.location='top_left'
    tab1 = Panel(child=p1, title="PM 2.5")
    

    tab1 = Panel(child=p1, title="PM 2.5")

tabs = Tabs(tabs=[ tab1])

show(tabs)
















