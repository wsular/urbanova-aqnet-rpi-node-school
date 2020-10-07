#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 21:12:22 2020

@author: matthew
"""



import numpy as np
import scipy
from bokeh.plotting import figure
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show



def linear_plot(x,y,unit_name):

    PlotType = 'HTMLfile'
    
    if PlotType=='notebook':
        output_notebook()
    else:
        #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_pad_resample.html')
        output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_mean_resample.html')
        

    #the data
   # x=np.array(df.Augusta)
   # y=np.array(df.Augusta)

    # determine best fit line (ie 1 to 1 line for avg. ref values)
    par = np.polyfit(x, x, 1, full=True)
    slope=par[0][0]
    intercept=par[0][1]
    y_predicted = [slope*i + intercept  for i in x]
    
    # For Paccar

    #the data for the Clarity node vs avg. ref value regression

    slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x, y)
    r_squared1 = r_value11**2

    # determine best fit line of the Clarity node vs avg. ref value regression
    par = np.polyfit(x, y, 1, full=True)
    slope1=par[0][0]
    intercept1=par[0][1]
    y_predicted1 = [slope1*i + intercept1  for i in x]

    # Mean Error Calc and performance stats
    print('\n')
    print('Ref_Avg average = ', x.mean(), '\n')
    print('Ref_Avg median = ', x.median(), '\n')
    print('Ref_Avg sum = ', x.sum(), '\n')
   
    print(unit_name + ' Adj average = ', y.mean(), '\n')
    print(unit_name + ' Adj median = ', y.median(), '\n')
    print(unit_name + ' Adj sum = ', y.sum(), '\n')
    
    numerator = ((y-x)**2).sum()
    print(numerator)
    print(type(numerator))
    numerator = float(numerator)
    print(type(numerator))
    denominator = len(y)
    print(denominator)
    print(type(denominator))
    denominator = float(denominator)
    print(type(denominator))
    print(numerator/denominator)
    rmse = (numerator/denominator)**0.5
    
    print(unit_name +  ' Adj RMSE = ', rmse, '\n')
    
    #print(unit_name +  ' Adj RMSE = ', ((y-x**2).sum()/len(y))**0.5, '\n')
    
    mae = (abs(x-y).sum())/(x.count())
        
    print(unit_name + ' mean absolute error =', mae, '\n')


    # plot it
    p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Clarity Nodes (ug/m^3)')

    p1.circle(x,x,legend='Ref Avg. 1 to 1 line', color='red')
    p1.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))

    p1.circle(x, y, legend=unit_name, color='blue')
    p1.line(x,y_predicted1,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))


    p1.legend.location='top_left'
   # p1.toolbar.logo = None
   # p1.toolbar_location = None
    
    tab1 = Panel(child=p1, title = 'Smoke Event Raw Data')



    tabs = Tabs(tabs=[ tab1])

    show(tabs)