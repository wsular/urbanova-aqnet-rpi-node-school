#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:34:21 2020

@author: matthew
"""
import pandas as pd
from glob import glob
import matplotlib as plt
#%%

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
    
#Read in SRCAA Augusta site BAM data

Augusta_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)
    
#%%
    
# Choose dates of interest
start_time = '2019-12-17 15:00'
end_time = '2020-01-20 23:00'

interval = '60T'
#%%

#Compare Clarity Units to Augusta SRCAA BAM as time series
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure


Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time]

Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar = Paccar_All.loc[start_time:end_time]
Paccar = Paccar.resample(interval).pad()


Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]
Reference = Reference.resample(interval).pad()



PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_overlap.html')
    
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')
        
p1.line(Augusta.index,     Augusta.PM2_5,  legend='Augusta',       color='green',     line_width=2)
p1.line(Paccar.index,     Paccar.PM2_5,  legend='Paccar',       color='blue',     line_width=2)
p1.line(Reference.index,     Reference.PM2_5,  legend='Reference',       color='red',     line_width=2)

tab1 = Panel(child=p1, title="Augusta BAM and Clarity Comparison")

tabs = Tabs(tabs=[ tab1])

show(tabs)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Compare Clarity Units to Augusta SRCAA BAM as time series


import pandas as pd

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy 
from bokeh.layouts import column
import holoviews as hv
import numpy as np
from bokeh.io import export_png

df = pd.DataFrame()
df['time'] = Augusta['time']
df['Augusta'] = Augusta['PM2_5']
df['Reference'] = Reference['PM2_5']
df['Paccar'] = Paccar['PM2_5']
df = df.dropna()


PlotType = 'HTMLfile'

if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter.html')

PlotType = 'HTMLfile'

#the data
x=np.array(df.Augusta)
y=np.array(df.Augusta)

# determine best fit line
par = np.polyfit(x, y, 1, full=True)
slope=par[0][0]
intercept=par[0][1]
y_predicted = [slope*i + intercept  for i in x]

#the data
x1=np.array(df.Augusta)
y1=np.array(df.Paccar) 
slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x1, y1)
r_squared1 = r_value11**2
# determine best fit line
par = np.polyfit(x1, y1, 1, full=True)
slope1=par[0][0]
intercept1=par[0][1]
y1_predicted = [slope1*i + intercept1  for i in x1]

#the data
x2=np.array(df.Augusta)
y2=np.array(df.Reference)
slope22, intercept22, r_value22, p_value22, std_err22 = scipy.stats.linregress(x2, y2)
r_squared2 = r_value22**2

# determine best fit line
par = np.polyfit(x2, y2, 1, full=True)
slope2=par[0][0]
intercept2=par[0][1]
y2_predicted = [slope2*i + intercept2  for i in x2]

# plot it
p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Clarity Nodes(ug/m^3)')

p1.circle(x,y,legend='BAM 1 to 1 line', color='red')
p1.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

p1.circle(df.Augusta, df.Paccar, legend='Paccar', color='blue')
p1.line(x1,y1_predicted,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))

p1.circle(df.Augusta, df.Reference, legend='Reference', color='green')
p1.line(x2,y2_predicted,color='green',legend='y='+str(round(slope2,2))+'x+'+str(round(intercept2,2))+ '  ' + 'r^2 = ' + str(round(r_squared2,3)))

p1.legend.location='top_left'

export_png(p1, filename= '/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_vs_wired_clarity_scatter.png')

tab1 = Panel(child=p1, title="SRCAA BAM vs Clarity Comparison")

tabs = Tabs(tabs=[ tab1])

show(tabs)



#%%
import numpy as np
import holoviews as hv

data = Paccar['PM2_5']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p2 = figure(plot_width = 1500,
            plot_height = 700)

p2 = hv.Histogram((edges, frequencies))
p2 = p2.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'Paccar')

data = Reference['PM2_5']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p3 = figure(plot_width = 1500,
            plot_height = 700)

p3 = hv.Histogram((edges, frequencies))
p3 = p3.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'Reference')

data = Augusta['PM2_5']
data = data.values
data = data[~np.isnan(data)]

frequencies, edges = np.histogram(data, 70)

p4 = figure(plot_width = 1500,
            plot_height = 700)

p4 = hv.Histogram((edges, frequencies))
p4 = p4.options(xlabel='PM 2.5 (ug/m3)', ylabel='Frequency', title = 'SRCAA BAM')

p5 = (p2+p3+p4).cols(3)

hv.save(p5.options(toolbar=None), '/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/histogram' 
        + start_time + '_to_' 
        + end_time + '.png', fmt='png', backend='bokeh')    # works

show(hv.render(p5))
#%%
p2 = figure(plot_width=900,
            plot_height=450,
            title='PM 2.5 Observations')

frequencies, edges = np.histogram(Paccar['PM2_5'],20)
p2 = hv.Histogram((edges, frequencies))




#%%
df = pd.DataFrame()
df['time'] = Augusta['time']
df['Augusta'] = Augusta['PM2_5']
df['Reference'] = Reference['PM2_5']
df['Paccar'] = Paccar['PM2_5']

from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ax = df.plot(kind="scatter", x="Augusta",y="Augusta", color="b", label="1 to 1 line")
df.plot(kind = 'scatter', x="Augusta",y="Paccar", color="r", label="BAM vs Paccar", ax=ax)
df.plot(kind = 'scatter', x="Augusta",y="Reference", color="g", label="BAM vs Reference", ax=ax)
ax.set_title('BAM vs Clarity Comparison')
ax.set_xlabel("BAM (ug/m3)")
ax.set_ylabel("Clarity Nodes (ug/m3)")
plt.show()