#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:41:57 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:35:16 2020

@author: matthew
"""
import pandas as pd
from glob import glob
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from bokeh.io import export_png
import numpy as np
import scipy 
from bokeh.layouts import row
from bokeh.layouts import gridplot

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
    
# Load EBAM Data
EBAM_data = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/calibration/Reference/EBAM*.csv')
files.sort()
for file in files:
    EBAM_data = pd.concat([EBAM_data, pd.read_csv(file)], sort=False)

EBAM_data.rename(columns={'ConcHR(ug/m3)':'PM2_5'}, inplace=True)
EBAM_data.rename(columns={'BP(mmHg)':'P'}, inplace=True)
EBAM_data.rename(columns={'RH(%)':'RH'}, inplace=True)
EBAM_data.rename(columns={'AT(C)':'Temp'}, inplace=True)

#convert EBAM Pressure from mmHg to mb 
EBAM_data.loc[:,'P'] *= 1.33322

#%%

start_time = '2019-07-01 00:00'
end_time = '2019-07-31 23:00'

interval = '60T'
#%%

### Batch 1 scatters and time series



Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]
Reference = Reference.resample(interval).mean()
    
Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar = Paccar_All.loc[start_time:end_time]
Paccar = Paccar.resample(interval).mean()

Audubon_All['time'] = pd.to_datetime(Audubon_All['time'])
Audubon_All = Audubon_All.sort_values('time')
Audubon_All.index = Audubon_All.time
Audubon = Audubon_All.loc[start_time:end_time]   
Audubon = Audubon.resample(interval).mean()

Balboa_All['time'] = pd.to_datetime(Balboa_All['time'])
Balboa_All = Balboa_All.sort_values('time')
Balboa_All.index = Balboa_All.time
Balboa = Balboa_All.loc[start_time:end_time]    
Balboa = Balboa.resample(interval).mean()

Browne_All['time'] = pd.to_datetime(Browne_All['time'])
Browne_All = Browne_All.sort_values('time')
Browne_All.index = Browne_All.time
Browne = Browne_All.loc[start_time:end_time]
Browne = Browne.resample(interval).mean()

Lidgerwood_All['time'] = pd.to_datetime(Lidgerwood_All['time'])
Lidgerwood_All = Lidgerwood_All.sort_values('time')
Lidgerwood_All.index = Lidgerwood_All.time
Lidgerwood = Lidgerwood_All.loc[start_time:end_time]
Lidgerwood = Lidgerwood.resample(interval).mean()

Regal_All['time'] = pd.to_datetime(Regal_All['time'])
Regal_All = Regal_All.sort_values('time')
Regal_All.index = Regal_All.time
Regal = Regal_All.loc[start_time:end_time]
Regal = Regal.resample(interval).mean()


EBAM_data['Time'] = pd.to_datetime(EBAM_data['Time'])
EBAM_data.index = EBAM_data.Time
#EBAM_All = EBAM_All.sort_values('time')
EBAM = EBAM_data.loc[start_time:end_time]
#EBAM = EBAM.resample(interval).mean()
EBAM = EBAM[EBAM['PM2_5'] < 300]   #remove extremely high EBAM values

#%%
PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/calibration/Clarity_wired_overlap_vs_EBAM_mean_resample.html')
    #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_overlap_pad_resample.html')
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(EBAM.index,      EBAM.PM2_5,    legend='EBAM',       color='gray',     line_width=2)
p1.line(Paccar.index,     Paccar.PM2_5,  legend='Wired 2',       color='blue',     line_width=2)
p1.line(Reference.index,     Reference.PM2_5,  legend='Reference',       color='red',     line_width=2)
p1.line(Audubon.index,     Audubon.PM2_5,  legend='#1',       color='yellow',     line_width=2)
p1.line(Balboa.index,     Balboa.PM2_5,  legend='#2',       color='green',     line_width=2)
p1.line(Browne.index,     Browne.PM2_5,  legend='#3',       color='purple',     line_width=2)
p1.line(Lidgerwood.index,     Lidgerwood.PM2_5,  legend='#4',       color='orange',     line_width=2)
p1.line(Regal.index,     Regal.PM2_5,  legend='#5',       color='brown',     line_width=2)

p1.legend.location='top_left'

tab1 = Panel(child=p1, title="Clarity Nodes and EBAM")


# tab for plotting scatter plots of clarity nodes vs clarity "Reference" wired node

df = pd.DataFrame()
df['Reference'] = Reference['PM2_5']
df['Audubon'] = Audubon['PM2_5']
df['Paccar'] = Paccar['PM2_5']
df['Balboa'] = Balboa['PM2_5']
df['Browne'] = Browne['PM2_5']
df['Lidgerwood'] = Lidgerwood['PM2_5']
df['Regal'] = Regal['PM2_5']
df['EBAM'] = EBAM['PM2_5']
df = df.dropna()

#the data for EBAM 1 to 1 line
x=np.array(df.EBAM)
y=np.array(df.EBAM)

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

#the data for EBAM vs Audubon
x1=np.array(df.EBAM)
y1=np.array(df.Audubon) 
slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x1, y1)
r_squared1 = r_value11**2
# determine best fit line
par = np.polyfit(x1, y1, 1, full=True)
slope1=par[0][0]
intercept1=par[0][1]
y1_predicted = [slope1*i + intercept1  for i in x1]

#the data for EBAM vs Balboa
x2=np.array(df.EBAM)
y2=np.array(df.Balboa)
slope22, intercept22, r_value22, p_value22, std_err22 = scipy.stats.linregress(x2, y2)
r_squared2 = r_value22**2

# determine best fit line
par = np.polyfit(x2, y2, 1, full=True)
slope2=par[0][0]
intercept2=par[0][1]
y2_predicted = [slope2*i + intercept2  for i in x2]


#the data for EBAM vs Browne
x3=np.array(df.EBAM)
y3=np.array(df.Browne)
slope33, intercept33, r_value33, p_value33, std_err33 = scipy.stats.linregress(x3, y3)
r_squared3 = r_value33**2

# determine best fit line
par = np.polyfit(x3, y3, 1, full=True)
slope3=par[0][0]
intercept3=par[0][1]
y3_predicted = [slope3*i + intercept3  for i in x3]


#the data for EBAM vs Lidgerwood
x4=np.array(df.EBAM)
y4=np.array(df.Lidgerwood)
slope44, intercept44, r_value44, p_value44, std_err44 = scipy.stats.linregress(x4, y4)
r_squared4 = r_value44**2

# determine best fit line
par = np.polyfit(x4, y4, 1, full=True)
slope4=par[0][0]
intercept4=par[0][1]
y4_predicted = [slope4*i + intercept4  for i in x4]

#the data for EBAM vs Regal
x5=np.array(df.EBAM)
y5=np.array(df.Regal)
slope55, intercept55, r_value55, p_value55, std_err55 = scipy.stats.linregress(x5, y5)
r_squared5 = r_value55**2

# determine best fit line
par = np.polyfit(x5, y5, 1, full=True)
slope5=par[0][0]
intercept5=par[0][1]
y5_predicted = [slope5*i + intercept5  for i in x5]


#the data for EBAM vs Paccar
x6=np.array(df.EBAM)
y6=np.array(df.Paccar)
slope66, intercept66, r_value66, p_value66, std_err66 = scipy.stats.linregress(x6, y6)
r_squared6 = r_value66**2

# determine best fit line
par = np.polyfit(x6, y6, 1, full=True)
slope6=par[0][0]
intercept6=par[0][1]
y6_predicted = [slope6*i + intercept6  for i in x6]

#the data for EBAM vs Reference
x7=np.array(df.EBAM)
y7=np.array(df.Reference)
slope77, intercept77, r_value77, p_value77, std_err77 = scipy.stats.linregress(x7, y7)
r_squared7 = r_value77**2

# determine best fit line
par = np.polyfit(x7, y7, 1, full=True)
slope7=par[0][0]
intercept7=par[0][1]
y7_predicted = [slope7*i + intercept7  for i in x7]

# plot it
p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Clarity Nodes(ug/m^3)')

p2.circle(x,y,legend='EBAM 1 to 1 line', color='red')
p2.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

#p2.circle(df.Reference, df.Audubon, legend='Audubon', color='blue')
#p2.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

#p2.circle(df.Reference, df.Balboa, legend='Balboa', color='green')
#p2.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))

#p2.circle(df.Reference, df.Browne, legend='Browne', color='yellow')
#p2.line(x3,y3_predicted,color='yellow',legend='y='+str(round(slope3,2))+'x+'+str(round(intercept3,2))+ '  ' + 'r^2 = ' + str(round(r_squared3,3)))

#p2.circle(df.Reference, df.Lidgerwood, legend='Lidgerwood', color='brown')
#p2.line(x4,y4_predicted,color='brown',legend='y='+str(round(slope4,2))+'x+'+str(round(intercept4,2))+ '  ' + 'r^2 = ' + str(round(r_squared4,3)))

#p2.circle(df.Reference, df.Regal, legend='Regal', color='purple')
#p2.line(x5,y5_predicted,color='purple',legend='y='+str(round(slope5,2))+'x+'+str(round(intercept5,2))+ '  ' + 'r^2 = ' + str(round(r_squared5,3)))

#p2.circle(df.Reference, df.Paccar, legend='Paccar', color='magenta')
#p2.line(x6,y6_predicted,color='magenta',legend='y='+str(round(slope6,2))+'x+'+str(round(intercept6,2))+ '  ' + 'r^2 = ' + str(round(r_squared6,3)))

p2.legend.location='top_left'
p2.toolbar.logo = None
#p2.toolbar_location = None


p3 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Audubon (ug/m^3)')

p3.circle(df.EBAM, df.Audubon, legend='Audubon', color='blue')
p3.legend.label_text_font_size = "10px"
p3.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

p3.legend.location='top_left'
p3.toolbar.logo = None
#p2.toolbar_location = None

p4 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Balboa (ug/m^3)')

p4.circle(df.EBAM, df.Balboa, legend='Balboa', color='green')
p4.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))
p4.legend.label_text_font_size = "10px"

p4.legend.location='top_left'
p4.toolbar.logo = None
#p2.toolbar_location = None

p5 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Browne (ug/m^3)')

p5.circle(df.EBAM, df.Browne, legend='Browne', color='gold')
p5.line(x3,y3_predicted,color='gold',legend='y='+str(round(slope3,2))+'x+'+str(round(intercept3,2))+ '  ' + 'r^2 = ' + str(round(r_squared3,3)))
p5.legend.label_text_font_size = "10px"

p5.legend.location='top_left'
p5.toolbar.logo = None
#p5.toolbar_location = None

p6 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Lidgerwood (ug/m^3)')

p6.circle(df.EBAM, df.Lidgerwood, legend='Lidgerwood', color='brown')
p6.line(x4,y4_predicted,color='brown',legend='y='+str(round(slope4,2))+'x+'+str(round(intercept4,2))+ '  ' + 'r^2 = ' + str(round(r_squared4,3)))
p6.legend.label_text_font_size = "10px"

p6.legend.location='top_left'
p6.toolbar.logo = None
#p6.toolbar_location = None

p7 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM(ug/m^3)',
            y_axis_label='Regal (ug/m^3)')

p7.circle(df.EBAM, df.Regal, legend='Regal', color='purple')
p7.line(x5,y5_predicted,color='purple',legend='y='+str(round(slope5,2))+'x+'+str(round(intercept5,2))+ '  ' + 'r^2 = ' + str(round(r_squared5,3)))
p7.legend.label_text_font_size = "10px"

p7.legend.location='top_left'
p7.toolbar.logo = None
#p7.toolbar_location = None

p8 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Paccar (ug/m^3)')

p8.circle(df.EBAM, df.Paccar, legend='Paccar', color='teal')
p8.line(x6,y6_predicted,color='teal',legend='y='+str(round(slope6,2))+'x+'+str(round(intercept6,2))+ '  ' + 'r^2 = ' + str(round(r_squared6,3)))
p8.legend.label_text_font_size = "10px"

p8.legend.location='top_left'
p8.toolbar.logo = None
#p7.toolbar_location = None

p9 = gridplot([[p3,p4, p5], [p6, p7, p8]], plot_width = 400, plot_height = 300)


tab2 = Panel(child=p9, title="Clarity Scatter Comparison vs EBAM")


# add in pressure so can make sure time series line up (easy to see)


p10 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (C)',
            y_axis_label='Paccar (C)')


p10.line(EBAM.index,EBAM.Temp,color='teal',legend='EBAM')
p10.line(Paccar.index,Paccar.temp,color='red',legend='Paccar')
p10.legend.label_text_font_size = "10px"

p10.legend.location='top_left'
p10.toolbar.logo = None

p11 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='EBAM (ug/m^3)',
            y_axis_label='Reference (ug/m^3)')

p11.circle(df.EBAM, df.Reference, legend='Reference', color='teal')
p11.line(x7,y7_predicted,color='teal',legend='y='+str(round(slope7,2))+'x+'+str(round(intercept7,2))+ '  ' + 'r^2 = ' + str(round(r_squared7,3)))
p11.legend.label_text_font_size = "10px"

p11.legend.location='top_left'
p11.toolbar.logo = None
#p7.toolbar_location = None

p9 = gridplot([[p3,p4, p5], [p6, p7, p8]], plot_width = 400, plot_height = 300)


tab3 = Panel(child=p10, title="Temp")

tab4 = Panel(child=p11, title="Reference")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4])


show(tabs)

export_png(p1, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_wired_time_series_mean_resample_vs_EBAM.png")
export_png(p9, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_scatter_mean_resample_vs_EBAM.png")
export_png(p11, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_scatter_mean_resample_vs_EBAM_reference.png")
#export_png(p1, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_wired_time_series_pad_resample_vs_EBAM.png")
#export_png(p9, filename="/Users/matthew/Desktop/data/calibration/Clarity_batch_1_scatter_pad_resample_vs_EBAM.png")

    
    
    
    
    
    
    
    
    
    
    
    
    