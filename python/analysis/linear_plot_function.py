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
import shapely
from shapely.geometry import LineString, Point
from bokeh.models import Label



def linear_plot(x,y,x_winter,y_winter,unit_name,n_lines,**kwargs):

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
    
    
    residuals_check = kwargs.get('residuals_check', None)
    
    if residuals_check == 1:
        residuals = kwargs.get('residuals', None)
        res_over_5 = abs(residuals).values
        res_over_5 = res_over_5[res_over_5 >= 5]
        
        count_over_5 = len(res_over_5)
        
        total_count = len(residuals)
        
        fraction_over = count_over_5/total_count
        fraction_under = 1 - fraction_over
        print(unit_name + ' Percentage of residuals over 5 ug/m3 = ', fraction_over)
        print(unit_name + ' Percentage of residuals under 5ug/m3 = ', fraction_under)

    # plot it
    p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='BAM (ug/m^3)',
            y_axis_label='Clarity Nodes (ug/m^3)')

    p1.circle(x,x,legend='Ref Avg. 1 to 1 line', color='red')
    p1.line(x,y_predicted,color='red',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)), line_width=3)

    p1.circle(x, y, legend=unit_name, color='blue')
    p1.line(x,y_predicted1,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)), line_width=3)

    
    # create extrapolatd data to find intersection of 2 calibration regimes
    x_extrap = np.linspace(0, 420, num=10)
    y_predicted_extrap_high = [slope1*i + intercept1  for i in x_extrap]
    
    #print(type(y_predicted_extrap_high))
    p1.line(x_extrap,y_predicted_extrap_high,color='blue', line_width=3)

    


    if n_lines ==  2:
        # determine best fit line (ie 1 to 1 line for avg. ref values)
        par = np.polyfit(x_winter, x_winter, 1, full=True)
        slope=par[0][0]
        intercept=par[0][1]
        y_predicted = [slope*i + intercept  for i in x_winter]
        
        # For Paccar
        
        #the data for the Clarity node vs avg. ref value regression
        
        slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x_winter, y_winter)
        r_squared1 = r_value11**2
        
        # determine best fit line of the Clarity node vs avg. ref value regression
        par = np.polyfit(x_winter, y_winter, 1, full=True)
        slope1=par[0][0]
        intercept1=par[0][1]
        y_predicted1 = [slope1*i + intercept1  for i in x_winter]
        
        # Mean Error Calc and performance stats
        print('\n')
        print('Ref_Avg average = ', x_winter.mean(), '\n')
        print('Ref_Avg median = ', x_winter.median(), '\n')
        print('Ref_Avg sum = ', x_winter.sum(), '\n')
        
        print(unit_name + ' Adj average = ', y_winter.mean(), '\n')
        print(unit_name + ' Adj median = ', y_winter.median(), '\n')
        print(unit_name + ' Adj sum = ', y_winter.sum(), '\n')
        
        numerator = ((y_winter-x_winter)**2).sum()
        print(numerator)
        print(type(numerator))
        numerator = float(numerator)
        print(type(numerator))
        denominator = len(y_winter)
        print(denominator)
        print(type(denominator))
        denominator = float(denominator)
        print(type(denominator))
        print(numerator/denominator)
        winter_rmse = (numerator/denominator)**0.5
        
        print(unit_name +  ' Adj RMSE = ', rmse, '\n')
        
        print(unit_name +  ' Adj RMSE = ', ((y_winter-x**2).sum()/len(y_winter))**0.5, '\n')
        
        winter_mae = (abs(x_winter-y).sum())/(x_winter.count())
            
        print(unit_name + ' mean absolute error =', mae, '\n')
        
        p1.circle(x_winter,x_winter,legend='Ref Avg. 1 to 1 line', color='red')
        p1.line(x_winter,y_predicted,color='red', line_width=3)#,legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
        
        p1.circle(x_winter, y_winter, legend=unit_name, color='blue')
        p1.line(x_winter,y_predicted1,color='green',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)), line_width=3)

        # create extrapolated data to connect the upper and lower regimes for the 1 to 1 reference line

        x_extrap_ref = np.linspace(0, 420, num=10)
        y_predicted_extrap_ref = [slope*i + intercept  for i in x_extrap_ref]
    
        p1.line(x_extrap_ref,y_predicted_extrap_ref,color='red', line_width=3)

        # create extrapolatd data to find intersection of 2 calibration regimes
        x_extrap_winter = np.linspace(0, 100, num=10)
        y_predicted_extrap_winter = [slope1*i + intercept1  for i in x_extrap_winter]
    
        p1.line(x_extrap_winter,y_predicted_extrap_winter,color='green', line_width=3)
        
        
        # find intersection of the two lines to determine where to switch claibration regimes
        # Given these endpoints
        
        #line 1
        A = (0, y_predicted_extrap_high[0])
        B = (x_extrap[-1], y_predicted_extrap_high[-1])
        
        #line 2
        C = (0, y_predicted_extrap_winter[0])
        D = (x_extrap_winter[-1], y_predicted_extrap_winter[-1])
        line1 = LineString([A, B])
        line2 = LineString([C, D])
        
        int_pt = line1.intersection(line2)
        #point_of_intersection = int_pt.x, int_pt.y

        ###convert from float to string so can use join in mytext label
        #intersect_string = tuple(str(round(x,2)) for x in point_of_intersection)
        #text = ','.join(intersect_string)
        #print(intersect_string)
        #print('Point of intersection: ' , point_of_intersection)
       ## #print(type(point_of_intersection))
        #mytext = Label(x=300, y=150, text='Intersection = (' + text + ')', text_font_size="10pt")

        #p1.add_layout(mytext)
        
        

    else:
        pass


    p1.legend.location='top_left'
   # p1.toolbar.logo = None
   # p1.toolbar_location = None
    
    tab1 = Panel(child=p1, title = 'Smoke Event Raw Data')



    tabs = Tabs(tabs=[ tab1])

    show(tabs)