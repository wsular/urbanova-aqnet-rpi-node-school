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
import statsmodels.api as sm
from bokeh.io import export_png, output_file
from bokeh.models import ranges
from power_law import power_law, power_law_cal
from scipy.optimize import curve_fit
import pandas as pd

def power_law(x, a, b):
    return a*np.power(x, b)
def power_law_cal(y, a , b):
    return (y/a)**(1/b)

def linear_plot(x,y,x_winter,y_winter,ref_avg,sensor_combined,unit_name,n_lines,unit_number,predictions_fire,predictions_winter,**kwargs):
    
    #print(sensor_combined)
    #print('ref_avg', ref_avg)
    combined_df = pd.DataFrame()
    combined_df['ref_avg'] = ref_avg
    combined_df['Clarity'] = sensor_combined
    combined_df = combined_df[combined_df['ref_avg'] > 0]

    
    xdata = combined_df[['ref_avg']].to_numpy()
    xdata = xdata[:, 0]
    #print(type(xdata))
    ydata = combined_df[['Clarity']].to_numpy()
    ydata = ydata[:, 0]
    #print(ydata)
    
    # Fit the dummy power-law data
    pars, cov = curve_fit(f=power_law, xdata=xdata, ydata=ydata, p0=[0, 0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the power calibrated data
    combined_df['power_fit_curve'] = power_law(xdata, *pars)
    combined_df['power_fit'] = power_law_cal(ydata, *pars)
    
    # print report with results and fitting statistics
   # print(type(pars))
    a = pars[0]
    b = pars[1]
    print('a,b = ', pars)
    
    lower_residuals = abs(combined_df['power_fit'] - combined_df['ref_avg'])
    lower_residuals_not_abs = (combined_df['power_fit'] - combined_df['ref_avg'])
    
  #  print(type(lower_residuals))
    print('stdev = ', lower_residuals_not_abs.std())
    
    lower_residuals = lower_residuals[lower_residuals >= 5]

    count_over_5 = len(lower_residuals)
    print('count_over_5 = ', count_over_5)
        
    total_count = len(combined_df['power_fit'])
    print('total count = ', total_count)
    fraction_over_power = count_over_5/total_count
    fraction_under_power = 1 - fraction_over_power
    print('Percentage of residuals over 5 ug/m3 (power) = ', fraction_over_power)
    print('Percentage of residuals under 5 ug/m3 (power) = ', fraction_under_power)
   
    # Mean Error Calc and performance stats
  #  print('\n')
  #  print('Ref_Avg average = ', combined_df['power_fit'].mean(), '\n')
  #  print('Ref_Avg median = ', combined_df['power_fit'].median(), '\n')
  #  print('Ref_Avg sum = ', combined_df['power_fit'].sum(), '\n')
   
  #  print(unit_name + ' Adj average = ', ref_avg.mean(), '\n')
  #  print(unit_name + ' Adj median = ', ref_avg.median(), '\n')
  #  print(unit_name + ' Adj sum = ', ref_avg.sum(), '\n')
    
    numerator = ((combined_df['ref_avg']-combined_df['power_fit'])**2).sum()
   # print('numerator', numerator)
   # print(type(numerator))
    numerator = float(numerator)
   # print(type(numerator))
    denominator = len(combined_df['ref_avg'])
   # print('denominator',denominator)
   # print(type(denominator))
    denominator = float(denominator)
   # print(type(denominator))
   # print(numerator/denominator)
    rmse_power = (numerator/denominator)**0.5
    
    print(unit_name +  ' Adj RMSE = ', rmse_power, '\n')
    
    #print(unit_name +  ' Adj RMSE = ', ((y-x**2).sum()/len(y))**0.5, '\n')
    
    mae_power = (abs(combined_df['power_fit']-combined_df['ref_avg']).sum())/(combined_df['power_fit'].count())
        
    print(unit_name + ' mean absolute error =', mae_power, '\n')
    
    mbe_power = (combined_df['power_fit']-combined_df['ref_avg']).sum()/(combined_df['power_fit'].count())
    
    
    lower_combined_df = pd.DataFrame()
    lower_combined_df['winter_predictions'] = combined_df['power_fit']
    lower_combined_df['reference'] = combined_df['ref_avg']
    lower_combined_df = lower_combined_df[lower_combined_df['reference'] < 50]
    
    numerator = ((lower_combined_df['reference']-lower_combined_df['winter_predictions'])**2).sum()
    numerator = float(numerator)
    denominator = len(lower_combined_df['reference'])
    denominator = float(denominator)
    rmse_power_winter = (numerator/denominator)**0.5
    print(unit_name +  ' combined fit winter Adj RMSE = ', rmse_power_winter, '\n')
    mae_power_winter = (abs(lower_combined_df['winter_predictions']-lower_combined_df['reference']).sum())/(lower_combined_df['winter_predictions'].count())
    print('combined fit, winter  mae = ', mae_power_winter)
    mbe_power_winter = ((lower_combined_df['winter_predictions']-lower_combined_df['reference']).sum())/lower_combined_df['winter_predictions'].count()
    
    
    
    upper_combined_df = pd.DataFrame()
    upper_combined_df['wildfire_predictions'] = combined_df['power_fit']
    upper_combined_df['reference'] = combined_df['ref_avg']
    upper_combined_df = upper_combined_df[upper_combined_df['reference'] > 50]
    
    numerator = ((upper_combined_df['reference']-upper_combined_df['wildfire_predictions'])**2).sum()
    numerator = float(numerator)
    denominator = len(upper_combined_df['reference'])
    denominator = float(denominator)
    rmse_power_wildfire = (numerator/denominator)**0.5
    print(unit_name +  ' combined fit wildfire Adj RMSE = ', rmse_power_wildfire, '\n')
    mae_power_wildfire = (abs(upper_combined_df['wildfire_predictions']-upper_combined_df['reference']).sum())/(upper_combined_df['wildfire_predictions'].count())
    print('combined fit, wildfire  mae = ', mae_power_wildfire)
    mbe_power_wildfire = ((upper_combined_df['wildfire_predictions']-upper_combined_df['reference']).sum())/upper_combined_df['wildfire_predictions'].count()
    
    PlotType = 'HTMLfile'
    
    if PlotType=='notebook':
        output_notebook()
    else:
        #output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_pad_resample.html')
        output_file('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/SRCAA_comparison_scatter_mean_resample.html')
        
    #the data
   # x=np.array(df.Augusta)
   # y=np.array(df.Augusta)

    X = x
    #X = X.dropna()
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

    # Note the difference in argument order
    model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
   # predictions = model.predict(y)
    
    # Print out the statistics
    print_model = model.summary()
 ###   print(print_model)

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
    
    lower_residuals = abs(predictions_fire - x)
    lower_residuals_not_abs = (predictions_fire - x)
    
  #  print(type(lower_residuals))
###    print('lower stdev = ', lower_residuals_not_abs.std())
    
    lower_residuals = lower_residuals[lower_residuals >= 5]

    count_over_5 = len(lower_residuals)
###    print('count_over_5 = ', count_over_5)
        
    total_count = len(predictions_fire)
###    print('total count = ', total_count)
    fraction_over = count_over_5/total_count
    fraction_under = 1 - fraction_over
###    print('Upper Percentage of residuals over 5 ug/m3 = ', fraction_over)
###    print('Upper Percentage of residuals under 5 ug/m3 = ', fraction_under)
   
    # Mean Error Calc and performance stats
###    print('\n')
###    print('Ref_Avg average = ', x.mean(), '\n')
###    print('Ref_Avg median = ', x.median(), '\n')
###    print('Ref_Avg sum = ', x.sum(), '\n')
   
###    print(unit_name + ' Adj average = ', y.mean(), '\n')
###    print(unit_name + ' Adj median = ', y.median(), '\n')
###    print(unit_name + ' Adj sum = ', y.sum(), '\n')
    
    numerator = ((x-predictions_fire)**2).sum()
   # print(numerator)
   # print(type(numerator))
    numerator = float(numerator)
   # print(type(numerator))
    denominator = len(x)
   # print(denominator)
   # print(type(denominator))
    denominator = float(denominator)
   # print(type(denominator))
   # print(numerator/denominator)
    rmse = (numerator/denominator)**0.5
    
    print(unit_name +  ' wildfire mlr Adj RMSE = ', rmse, '\n')
    
    #print(unit_name +  ' Adj RMSE = ', ((y-x**2).sum()/len(y))**0.5, '\n')
    
    mae = (abs(predictions_fire-x).sum())/(predictions_fire.count())
        
    print(unit_name + ' wildfire mean absolute error =', mae, '\n')
    
    mbe = ((predictions_fire-x).sum())/predictions_fire.count()
    print()
    
    residuals_check = kwargs.get('residuals_check', None)
    
    if residuals_check == 1:
        residuals = kwargs.get('residuals', None)
        res_over_5 = abs(residuals).values
        res_over_5 = res_over_5[res_over_5 >= 5]
        
        count_over_5 = len(res_over_5)
        
        total_count = len(residuals)
        
        fraction_over = count_over_5/total_count
        fraction_under = 1 - fraction_over
  ###      print(unit_name + ' Percentage of residuals over 5 ug/m3 = ', fraction_over)
  ###      print(unit_name + ' Percentage of residuals under 5ug/m3 = ', fraction_under)


    # plot it
    p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Reference Data (ug/m³)',
            y_axis_label= unit_number + ' (ug/m³)',
            y_range= ranges.Range1d(start=0,end=800), 
            x_range= ranges.Range1d(start=0,end=500) 
            )

    #p1.title.text = unit_number 
    #p1.title.text_font = "times"
    #p1.circle(x,x,legend='Ref Avg. 1 to 1 line', color='red')
    #p1.line(x,y_predicted,color='black',legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)), line_width=3, line_dash='dashed')

    p1.circle(x, y, color='black', size=5)#, legend='Wildfire: ' + 'y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r² = ' + str(round(r_squared1,3))+ ' ' + 'MAE = ' +str(round(mae,2)) + ' ' + 'RMSE =' + str(round(rmse,2)) + ' MBE = ' + str(round(mbe,2)))
   # p1.line(x,y_predicted1,color='blue',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)), line_width=3)

    
    # create extrapolatd data to find intersection of 2 calibration regimes
    x_extrap = np.linspace(0, 420, num=10)
    y_predicted_extrap_high = [slope1*i + intercept1  for i in x_extrap]
    
    #print(type(y_predicted_extrap_high))
    p1.line(x_extrap,y_predicted_extrap_high,color='black', line_width=3)

    


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
        
        X = x_winter
        #X = X.dropna()
        X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
        
        # Note the difference in argument order
        model = sm.OLS(y_winter, X).fit() ## sm.OLS(output, input)
     #   predictions_winter = model.predict(y_winter)
        
        
        
        par = np.polyfit(x_winter, y_winter, 1, full=True)
        slope1=par[0][0]
        intercept1=par[0][1]
        y_predicted1 = [slope1*i + intercept1  for i in x_winter]
        
        upper_residuals = abs(x_winter - predictions_winter)
        # need this for the stdev of the residuals for the uncertainty (as using abs changes this)
        upper_residuals_not_abs = x_winter - predictions_winter
       # print('123')
      #  print(x_winter)
      #  print('1')
     #  print(y_winter)
      #  print('2')
     #   print(upper_residuals)
      #  print(type(upper_residuals))
        print('upper stdev = ', upper_residuals_not_abs.std())
        
        
      
        upper_residuals = upper_residuals[upper_residuals >= 5]

        count_over_5 = len(upper_residuals)
 ###       print('count_over_5 = ', count_over_5)
    
        total_count = len(x_winter)
  ###      print('total count = ', total_count)
        fraction_over = count_over_5/total_count
        fraction_under = 1 - fraction_over
  ###      print('Upper Percentage of residuals over 5 ug/m3 = ', fraction_over)
  ###      print('Upper Percentage of residuals under 5 ug/m3 = ', fraction_under)
    
    # Mean Error Calc and performance stats
  ###      print('\n')
 ###       print('Ref_Avg average = ', x_winter.mean(), '\n')
 ###       print('Ref_Avg median = ', x_winter.median(), '\n')
 ###       print('Ref_Avg sum = ', x_winter.sum(), '\n')
        
   ###     print(unit_name + ' Adj average = ', y_winter.mean(), '\n')
  ###      print(unit_name + ' Adj median = ', y_winter.median(), '\n')
   ###     print(unit_name + ' Adj sum = ', y_winter.sum(), '\n')
        
        numerator = ((predictions_winter-x_winter)**2).sum()
     #   print(numerator)
     #   print(type(numerator))
        numerator = float(numerator)
     #   print(type(numerator))
        denominator = len(predictions_winter)
     #   print(denominator)
     #   print(type(denominator))
        denominator = float(denominator)
     #   print(type(denominator))
     #   print(numerator/denominator)
        winter_rmse = (numerator/denominator)**0.5
        
        print(unit_name +  ' winter Adj RMSE = ', winter_rmse, '\n')
        
    ###    print(unit_name +  ' winter Adj RMSE = ', ((y_winter-x_winter**2).sum()/len(y_winter))**0.5, '\n')
        
        winter_mae = (abs(x_winter-predictions_winter).sum())/(x_winter.count())
            
        print(unit_name + ' winter mean absolute error =', winter_mae, '\n')
        
        winter_mbe = (x_winter-predictions_winter).sum()/(x_winter.count())
        
        # 1 to 1 line already plotted above
        #p1.circle(x_winter,x_winter,legend='Ref Avg. 1 to 1 line', color='red')
        #p1.line(x_winter,y_predicted,color='black', line_width=2, line_dash='dashed')#,legend='y='+str(round(slope,2))+'x+'+str(round(intercept,2)))
        
        # lower regime data
        p1.triangle(x_winter, y_winter, color='gray', size=8)#, legend='Winter:' + ' y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r² = ' + str(round(r_squared1,3))+ ' ' + 'MAE = ' +str(round(winter_mae,2)) + ' ' + 'RMSE = ' + str(round(winter_rmse,2)) + ' MBE = ' + str(round(winter_mbe,2)))
        # this is just commented out so dont have two entries in legend (wanted to keep the extrapolated line for high regime just for visual purposes)
      #  p1.line(x_winter,y_predicted1,color='green',line_width=3)#, legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)), line_width=3)

        # create extrapolated data to connect the upper and lower regimes for the 1 to 1 reference line

        x_extrap_ref = np.linspace(0, 900, num=10)
        y_predicted_extrap_ref = [slope*i + intercept  for i in x_extrap_ref]
        
        # add the fit of the upper regime to the extrapolated data (so don't have too many things in legend)
        p1.line(x_extrap_ref,y_predicted_extrap_ref,color='gray', line_width=2, line_dash='dashed')#, legend = 'Power fit, Wildfire: MAE = ' + str(round(mae_power_wildfire,2)) + ' ' + 'RMSE = ' + str(round(rmse_power_wildfire,2)) + ' MBE = ' + str(round(mbe_power_wildfire,2)))

        # create extrapolatd data to find intersection of 2 calibration regimes
        x_extrap_winter = np.linspace(0, 900, num=10)
        y_predicted_extrap_winter = [slope1*i + intercept1  for i in x_extrap_winter]
    
        p1.line(x_extrap_winter,y_predicted_extrap_winter,color='gray', line_width=3)#, legend = 'Power fit, Winter: MAE = ' + str(round(mae_power_winter,2)) + ' ' + 'RMSE = ' + str(round(rmse_power_winter,2)) + ' MBE = ' + str(round(mbe_power_winter,2)))
        
        
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
        
        # create extrapolated data to plot powerfit line to raw combined data set

        x_extrap_ref = np.linspace(0, 500, num=500)
        y_predicted_extrap_ref = [a*i**b  for i in x_extrap_ref]
        
        #plots points for power law fit ## p1.scatter(xdata, power_law(xdata, *pars), legend='Combined Fit: ' + 'y=' + str(round(a,2)) + 'x^' + str(round(b,2)),       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
        p1.line(x_extrap_ref, y_predicted_extrap_ref,    color='green',       line_width=2, muted_color='green', muted_alpha=0.2)#, legend='Power Fit Combined: ' + 'y=' + str(round(a,2)) + 'x^' + str(round(b,2)) + ' ' + 'MAE = ' +str(round(mae_power,2)) + ' ' + 'RMSE = ' + str(round(rmse_power,2)) + ' MBE = ' + str(round(mbe_power,2)))
        

    else:
        pass

    p1.legend.location='top_left'
    p1.legend.label_text_font_size = "9pt"
    p1.legend.label_text_font = "times"
    p1.legend.label_text_color = "black"
    
   # p1.xaxis.axis_label="xaxis_name"
    p1.xaxis.axis_label_text_font_size = "14pt"
    p1.xaxis.major_label_text_font_size = "14pt"
    p1.xaxis.axis_label_text_font = "times"
    p1.xaxis.axis_label_text_color = "black"
    p1.xaxis.major_label_text_font = "times"

   # p1.yaxis.axis_label="yaxis_name"
    p1.yaxis.axis_label_text_font_size = "14pt"
    p1.yaxis.major_label_text_font_size = "14pt"
    p1.yaxis.axis_label_text_font = "times"
    p1.yaxis.axis_label_text_color = "black"
    p1.yaxis.major_label_text_font = "times"
    p1.toolbar.logo = None
    p1.toolbar_location = None
    
  
   # p1.xgrid.grid_line_color = None
   # p1.ygrid.grid_line_color = None
    
   # p1.x_range.range_padding = 0
   # p1.y_range.range_padding = 0
    
    # for audubon indoor cal raw data
   # export_png(p1,'/Users/matthew/Desktop/thesis/Final_Figures/Materials_and_Methods/Audubon_indoor_cal_raw_data.png')
    # for audubon indoor cal corrected data
   # export_png(p1,'/Users/matthew/Desktop/thesis/Final_Figures/Materials_and_Methods/Audubon_indoor_cal_corrected_data.png')
    # for audubon combined calibration raw data
    #export_png(p1,'/Users/matthew/Desktop/thesis/Final_Figures/Materials_and_Methods/Audubon_combined_cal_raw_data.png')
    tab1 = Panel(child=p1, title = 'Smoke Event Raw Data')

    

    tabs = Tabs(tabs=[ tab1])

    show(tabs)
    
    return p1#, upper_combined_df, lower_combined_df