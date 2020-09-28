#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:33:09 2020

@author: matthew
"""


import pandas as pd
from bokeh.layouts import gridplot
PlotType = 'HTMLfile'
import holoviews as hv
hv.extension('bokeh', logo=False)
import geoviews as gv
import geoviews.tile_sources as gvts
from bokeh.plotting import show

info = [[47.621172,   -117.367725,   24.47,  'Adams'],
        [47.621533,   -117.4098417,  47.23,  'Jefferson'],
        [47.6467083,  -117.390983,   60.48,  'Grant'],
        [47.6522472,  -117.355561,   79.46,  'Sheridan'],
        [47.6608,     -117.4045056,  37,     'Reference'],
        [47.671256,   -117.3846583,  42.3,   'Stevens'],
        [47.69735,    -117.369972,   26.94,  'Regal'],
        [47.6798472,  -117.441739,   18.52,  'Audubon'],
        [47.7081417,  -117.405161,   24.55,  'Lidgerwood'],
        [47.70415,    -117.4640639,  31.66,  'Browne'],
        [47.71818056, -117.4560056,  47.13,  'Balboa']]

df = pd.DataFrame(info, columns =['Lat', 'Lon', 'avg_PM2_5','Location'], dtype = float)
#print(df.dtypes)

label_locations = [[6044100,     -13062900,   'Adams'],
                  [6044200,      -13073000,   'Jefferson'],
                  [6048400,      -13070000,   'Grant'],
                  [6048100,      -13062300,   'Sheridan'],
                  [6050600,      -13073000,   'Reference'],
                  [6052300,      -13064300,   'Stevens'],
                  [6056700,      -13063400,   'Regal'],
                  [6053850,      -13076600,   'Audubon'],
                  [6059950,      -13068000,   'Lidgerwood'],
                  [6057850,      -13078600,   'Browne'],
                  [6061500,      -13075300,   'Balboa']]

df1 = pd.DataFrame(label_locations, columns =['Lat', 'Lon', 'Location'], dtype = float)
  


scatter = hv.Scatter(df.dropna(), kdims='Lon', vdims=['Lat', 'avg_PM2_5', 'Location'])
scatter.opts(color='avg_PM2_5', size=10, padding=.1, tools=['hover'], colorbar=True, cmap='magma', width=500, height=400, clim=(0, 60))

points = gv.Points(df.dropna(), ['Lon', 'Lat'], ['avg_PM2_5', 'Location'])
points.opts(size=10, color='avg_PM2_5', cmap='magma', tools=['hover'], colorbar=True, width=500, height=400, padding=.1, clim=(0, 60))

labels = hv.Labels(df1, kdims = ['Lon', 'Lat'], vdims =['Location'])

test = gvts.EsriImagery * points * labels
hv.save(test.options(toolbar=None), '/Users/matthew/Desktop/IDW_new_test.png', fmt='png', backend='bokeh')    # works
#show(hv.render(test))

