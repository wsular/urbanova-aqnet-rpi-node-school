#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:53:22 2020

@author: matthew
"""
import pandas as pd
from metpy.units import units
import metpy.calc as mpcalc
import metpy
#%%

def spec_humid(csv_data, json_data, Clarity_node):
# load in stevens pressure data (closest to Augusta site) for specific humidity calcs

  
    bme1 = pd.DataFrame({})
    bme2 = pd.DataFrame({})
    
    bme1['P'] = json_data['P']
    bme1.index = json_data.index

    bme2['P'] = csv_data['P']
    bme2.index = csv_data.index

    combined = bme1.append(bme2)

    Clarity_node['pressure'] = combined

    Clarity_node_pressure = []

    for row in Clarity_node['pressure']:
        Clarity_node_pressure.append(row*units.mbar)      ## NEED TO CHECK WHAT UNITS THIS SHOULD BE IN - checked, should be mbar

    #print(Paccar_pressure)

    ##This has been hand checked and looks good

    Clarity_node_temp = []

    for row in Clarity_node['temp']:
        Clarity_node_temp.append(row*units.degC)
    print(len(Clarity_node_temp))

    Clarity_node_rh = []

    for row in Clarity_node['Rel_humid']:
        Clarity_node_rh.append(row*units.percent)
    print(len(Clarity_node_rh))

    Clarity_node_dewpoint = []

    for (temp, rh) in list(zip(Clarity_node_temp, Clarity_node_rh)): 
        #print(metpy.calc.dewpoint_from_relative_humidity(temp, rh))
        Clarity_node_dewpoint.append(metpy.calc.dewpoint_from_relative_humidity(temp, rh))
        #print(Clarity_node_dewpoint)
        
    Clarity_node['dewpoint'] = Clarity_node_dewpoint
        
    #### For some reason only works if in reversed order from documentation (looks like a doc error because get units error when do the doc way)
    Clarity_node_spec_humid = []
        
    for (dewpoint, pressure) in list(zip(Clarity_node_dewpoint, Clarity_node_pressure)):
    #print(metpy.calc.specific_humidity_from_dewpoint(dewpoint,pressure))
        Clarity_node_spec_humid.append(metpy.calc.specific_humidity_from_dewpoint(dewpoint,pressure))
        #print(Clarity_node_spec_humid)

    Clarity_node['spec_humid'] = Clarity_node_spec_humid

    Clarity_node['spec_humid_unitless'] = Clarity_node['spec_humid'].astype(float) # Note that "unitless" refers to unitless within the df so can be plotted. the actual units at this point are kg/kg )also unitless but different intent than removing the "dimensionless" from the metpy numbers

    Clarity_node['spec_humid_unitless'] = Clarity_node['spec_humid_unitless']*1000 # units of g/kg
    
    return Clarity_node