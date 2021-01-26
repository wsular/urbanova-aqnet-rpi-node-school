#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:43:16 2020

@author: matthew
"""

import pandas as pd
from glob import glob
import numpy as np
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show

from bokeh.plotting import figure
import holoviews as hv
hv.extension('bokeh', logo=False)
from high_cal_mlr_function_generator import high_cal_setup, generate_mlr_function_high_cal
from high_cal_mlr_function import mlr_function_high_cal
from scipy.optimize import curve_fit
import datetime
import copy
from load_indoor_data import load_indoor
from linear_plot_function import linear_plot

# Function to calculate the power-law with constants a and b
def power_law(x, a, b):
    return a*np.power(x, b)
def power_law_cal(y, a , b):
    return (y/a)**(1/b)



def power_law_fit(name, final_cal_data_set):
    xdata = final_cal_data_set[['ref_value']].to_numpy()
    xdata = xdata[:, 0]
    ydata = final_cal_data_set[['indoor']].to_numpy()
    ydata = ydata[:, 0]

    # Fit the  power-law data
    pars, cov = curve_fit(f=power_law, xdata=xdata, ydata=ydata, p0=[0, 0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    residuals = ydata - power_law(xdata, *pars)
    # residual sum of squares
    ss_res = np.sum(residuals**2)
    # total sum of squares
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    r_squared = 1-(ss_res/ss_tot)
    a,b = pars
    print('a,b = ', pars)
    print('r^2 = ', r_squared)
    
    # add power fit (ie trendline) to df
    final_cal_data_set['Power_fit'] = power_law(xdata, *pars)
    
    # add calibrated data to df
    final_cal_data_set['Indoor_calibrated'] = power_law_cal(ydata, *pars)
    
    p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Selected Ref (ug/m3)',
            y_axis_label='Indoor PM 2.5 (ug/m3)')

    p1.title.text = 'Power Fit'
    # plot scatter of final calibration data
    p1.scatter(final_cal_data_set.ref_value,      final_cal_data_set.indoor,    legend=(name + ' Final'),     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
    p1.scatter(final_cal_data_set.ref_value,      final_cal_data_set.Power_fit,    legend='y = ' + str(round(a,2)) + 'x' + '^' + str(round(b,2)) + ' r^2 = ' + str(round(r_squared,2)),     color='red',       line_width=2, muted_color='olive', muted_alpha=0.2)

    p1.legend.click_policy="hide"
    tab1 = Panel(child=p1, title="Final Calibration Dataset")
    
    p2 = figure(plot_width=900,
            plot_height=450,
            x_axis_label='Selected Ref (ug/m3)',
            y_axis_label='Calibrated Indoor PM 2.5 (ug/m3)')

    p2.title.text = 'Calibrated Indoor Data'
    # plot scatter of final calibrated data
    p2.scatter(final_cal_data_set.ref_value,      final_cal_data_set.Indoor_calibrated,    legend=(name + ' Final'),     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
    p2.scatter(final_cal_data_set.ref_value,      final_cal_data_set.ref_value,    legend='1 to 1 line',     color='red',       line_width=2, muted_color='olive', muted_alpha=0.2)

    p2.legend.click_policy="hide"
    tab2 = Panel(child=p2, title="Calibrated Data (single power function)")
    
    tabs = Tabs(tabs=[ tab1, tab2])
    show(tabs)
    
    return final_cal_data_set


# function to create 100 sec averages of indoor data matched with Clarity node measurements

# variable inputs:
# dt = time delta of the Clarity node averaging period
# Ref_df = Clarity node being used as the Reference (Browne or Reference)
# indoor_df = the indoor unit being matche to the reference data

def data_match(dt, Ref_df, indoor_df):#, indoor_BME280_df):
    ref_times = pd.DataFrame({})
    ref_times['end'] = Ref_df['time']
    ref_times['start'] = ref_times['end'] - dt


    indoor_avg = []
   # indoor_temp = []
   # indoor_rh = []
   # indoor_p = []
  #  Reference_node_PM2_5_corrected = Ref_df['PM2_5_corrected'].tolist()


    for start,end in zip(ref_times.start, ref_times.end):
        start_pt = start
        #print(start_pt)
        #  print(type(start_pt))
        #  start_str = str(start_pt)
        #  print(start_str)
        #  print(type(start_str))
        end_pt = end
        #print(end_pt)
        #  print(type(end_pt))
        #  end_str = str(end_pt)
        #  print(type(end_str))
        #  print(end_str)
        
        # copy indoor_df so not modifying original dataframe
        indoor = indoor_df.copy()
        
        # replace PM_0_3 column (just used as placeholder, could have been any column but not using it for anything)
        # with flags to determine where the data is between the start and end points of the data collected for each
        # data point from the Ref_df Clarity node
        indoor['PM_0_3'] = np.where(indoor.Datetime > end_pt, 0, 
                                    (np.where(indoor.Datetime < start_pt, 0, 50)))
    
        # create df so that only the relevant measurements are included
        indoor = indoor[indoor.PM_0_3 != 0]
        #print(indoor)
        
        # take average of PM2.5 measurements for this range and append to new list
        PM2_5_avg = indoor['PM2_5_env'].mean()
        indoor_avg.append(PM2_5_avg)
        
        
        # add temp and rh data from Clarity node so that can use to make MLR's
      #  ref = Ref_df.copy()
        
     #   temp = ref.loc[end]['temp']
     #   rh = ref.loc[end]['Rel_humid']
        
     #   indoor_temp.append(temp)
     #   indoor_rh.append(rh)

    
    # combine lists into single df
        
    indoor_measurements_and_Ref = pd.DataFrame({'indoor_PM2_5_avg':indoor_avg, 
                                          #      'indoor_temp_avg':indoor_temp, 
                                          #      'indoor_rh_avg':indoor_rh,
                                                'Reference_node_PM2_5_corrected':Ref_df['PM2_5_corrected'].tolist()})
    indoor_measurements_and_Ref.index = Ref_df.time
    indoor_measurements_and_Ref['Time'] = Ref_df.time

    return indoor_measurements_and_Ref


# create final df after choosing which reference node the indoor data is compared to based on the thresholds for each regime
# go through each match df
#   - if ref measurement is below min for Ref, append to new df else skip
#   - if ref measurement is above min for Browne,append to new df else skip


def final_df_cal_data(ref_match_df, Browne_match_df):

    low_regime = []
    low_regime_ref_values = []
    low_regime_times = []
    
    mid_and_upper_regime = []
    mid_and_upper_regime_ref_values = []
    mid_and_upper_regime_times = []

    for indoor, ref, time in zip(ref_match_df.indoor_PM2_5_avg, ref_match_df.Reference_node_PM2_5_corrected, ref_match_df.Time):
        
        if indoor < 180:      # screens out any intial points that the ref unit didnt catch
            if ref < 68:
                low_regime.append(indoor)
                low_regime_ref_values.append(ref)
                low_regime_times.append(time)
            else:
                pass
        else:
            pass
        
    for indoor, ref, time in zip(Browne_match_df.indoor_PM2_5_avg, Browne_match_df.Reference_node_PM2_5_corrected, Browne_match_df.Time):
        
        if ref > 68:
            # remove the outlier from Audubon (found from visually inspecting time series of data)
            if round(indoor) == 867:
                pass
            # remove the outlier from Adams
            elif round(indoor) == 825:
                pass
            # remove outlier from Balboa
            elif round(indoor) == 878:
                pass
            # remove outlier from Browne
            elif round(indoor) == 967:
                pass
            # remove outlier from Grant
            elif round(indoor) == 839:
                pass
            # remove outlier from Jefferson
            elif round(indoor) == 741:
                pass
            # remove outlier from Lidgerwood
            elif round(indoor) == 921:
                pass
            # remove outlier from Regal
            elif round(indoor) == 791:
                pass
            # remove outlier from Sheridan
            elif round(indoor) == 813:
                pass
            # remove outlier from Stevens
            elif round(indoor) == 880:
                pass
            
            else:
                mid_and_upper_regime.append(indoor)
                mid_and_upper_regime_ref_values.append(ref)
                mid_and_upper_regime_times.append(time)
        else:
            pass

    # add upper regimes to lower (note that even though called low_regime, the lists have all the data from both)
    low_regime.extend(mid_and_upper_regime)
    low_regime_ref_values.extend(mid_and_upper_regime_ref_values)
    low_regime_times.extend(mid_and_upper_regime_times)
    final_df = pd.DataFrame({'indoor':low_regime, 'ref_value':low_regime_ref_values, 'time':low_regime_times})  
    final_df = final_df.sort_values('time')
    final_df.index = final_df.time
    
    return final_df


# inputs: the df that matches each indoor unit averaged measurements with the Reference node measurements
          # the df that matches each indoor unit averaged measurements with the Browne node measurements
def Indoor_cal_generator(ref_match_df, Browne_match_df):
    
    # select data for creating mlr for low-regime
    mlr_1_df = ref_match_df[ref_match_df['ref_avg'] < 68]
    
    y = ref_match_df['ref_avg']
    X = ref_match_df[['Audubon', 'Audubon_rh', 'Audubon_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

    model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    predictions = model.predict(X)

    # Print out the statistics
    print_model = model.summary()
    print(print_model)

    
    
    return mlr_1, power_fit, mlr_2


#%%

calibration_df = high_cal_setup()
mlr_high_browne = generate_mlr_function_high_cal(calibration_df, 'Browne')

# set time delta for Clarity measurement length 
dt = datetime.timedelta(seconds=100)

PlotType = 'HTMLfile'

ModelType = 'mlr'    # options: rf, mlr, hybrid, linear
stdev_number = 1   # defines whether using 1 or 2 stdev for uncertainty

slope_sigma1 = 2       # percent uncertainty of SRCAA BAM calibration to reference clarity slope
slope_sigma2 = 4.5     # percent uncertainty of slope for paccar roof calibrations (square root of VAR slope from excel)
slope_sigma_paccar = 2     # percent uncertainty of slope for Paccar Clarity unit at SRCAA BAM calibration
sigma_i = 5    

# Test day 1
#start_time = '2020-11-21 13:00'
#end_time = '2020-11-22 00:00'

# Test day 2
#start_time = '2020-12-01 09:00'
#end_time = '2020-12-01 12:00'

# Test day 3
#start_time = '2020-12-07 09:00'
#end_time = '2020-12-07 16:00'

# All data
start_time = '2020-11-21 13:00'  
end_time = '2020-12-07 16:00'

interval = '10S'    # for plotting indoor/outdoor comparisons
#interval = '60T'

calibration_df = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/combined_calibration/*.csv')
files.sort()
for file in files:
    calibration_df = pd.concat([calibration_df, pd.read_csv(file)], sort=False)

calibration_df['time'] = pd.to_datetime(calibration_df['time'])
calibration_df = calibration_df.sort_values('time')
calibration_df.index = calibration_df.time
del calibration_df['ref_stdev']
calibration_df = calibration_df.dropna()

Browne_df = pd.DataFrame()
Browne_df['ref_avg'] = calibration_df.ref_avg
Browne_df['Browne'] = calibration_df.Browne
Browne_df = Browne_df[Browne_df['ref_avg'] > 0]
#Browne_df.to_csv('/Users/matthew/Desktop/Browne_power_cal.csv', index=False)

xdata = Browne_df[['ref_avg']].to_numpy()
xdata = xdata[:, 0]
print(type(xdata))
ydata = Browne_df[['Browne']].to_numpy()
ydata = ydata[:, 0]


# Fit the  power-law data
pars, cov = curve_fit(f=power_law, xdata=xdata, ydata=ydata, p0=[0, 0], bounds=(-np.inf, np.inf))
# Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
stdevs = np.sqrt(np.diag(cov))
# Calculate the residuals
Browne_df['Browne_power_fit'] = power_law_cal(ydata, *pars)


Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv')
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)


Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]

Reference['PM2_5_corrected'] = (Reference['PM2_5'] + 0.6232)/1.7588   # From AUGUSTA BAM comparison

Browne_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Browne*.csv')
files.sort()
for file in files:
    Browne_All = pd.concat([Browne_All, pd.read_csv(file)], sort=False)

# drop erroneous data from Nov. 2019 when sensor malfunctioning
Browne_All = Browne_All[Browne_All['PM2_5'] < 1000]

Browne_All['time'] = pd.to_datetime(Browne_All['time'])
Browne_All = Browne_All.sort_values('time')
Browne_All.index = Browne_All.time
Browne = Browne_All.loc[start_time:end_time]

#Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 > 74), mlr_function_high_cal(mlr_high_browne, Browne), Browne.PM2_5)  # high calibration adjustment
#Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 < 74 & Browne.PM2_5 > 68), power_law_cal(ydata, *pars), Browne.PM2_5)  # power fit calibration adjustment

#Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 < 68), (Browne.PM2_5-0.4771)/1.1082, Browne.PM2_5_corrected)  # Paccar roof adjustment
#Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 < 68), Browne.PM2_5*0.454-Browne.Rel_humid*0.0483-Browne.temp*0.0774+4.8242, Browne.PM2_5_corrected)  # high calibration adjustment

Browne['PM2_5_corrected'] = np.where(Browne.PM2_5 > 74, mlr_function_high_cal(mlr_high_browne, Browne), 
         (np.where(Browne.PM2_5 < 68, (Browne.PM2_5-0.4771)/1.1082, power_law_cal(Browne.PM2_5, *pars))))
Browne['PM2_5_corrected'] = np.where((Browne.PM2_5 < 68), Browne.PM2_5*0.454-Browne.Rel_humid*0.0483-Browne.temp*0.0774+4.8242, Browne.PM2_5_corrected)  # Clarity ref to BAM adjustment

# /chamber only has data for the chamber test
# /indoor_cal/* has all the overlap data (takes a really long time to load)

#%%

grant = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Grant/indoor_cal/*.csv')
files.sort()
for file in files:
    grant = pd.concat([grant, pd.read_csv(file)], sort=False)


stevens = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Stevens/indoor_cal/*.csv')
files.sort()
for file in files:
    stevens = pd.concat([stevens, pd.read_csv(file)], sort=False)


balboa = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Balboa/indoor_cal/*.csv')
files.sort()
for file in files:
    balboa = pd.concat([balboa, pd.read_csv(file)], sort=False)


adams = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Adams/indoor_cal/*.csv')
files.sort()
for file in files:
    adams = pd.concat([adams, pd.read_csv(file)], sort=False)


jefferson = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Jefferson/indoor_cal/*.csv')
files.sort()
for file in files:
    jefferson = pd.concat([jefferson, pd.read_csv(file)], sort=False)


sheridan = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Sheridan/indoor_cal/*.csv')
files.sort()
for file in files:
    sheridan = pd.concat([sheridan, pd.read_csv(file)], sort=False)


browne = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Browne/indoor_cal/*.csv')
files.sort()
for file in files:
    browne = pd.concat([browne, pd.read_csv(file)], sort=False)


audubon = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Audubon/indoor_cal/*.csv')
files.sort()
for file in files:
    audubon = pd.concat([audubon, pd.read_csv(file)], sort=False)


lidgerwood = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Lidgerwood/indoor_cal/*.csv')
files.sort()
for file in files:
    lidgerwood = pd.concat([lidgerwood, pd.read_csv(file)], sort=False)


regal = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Regal/indoor_cal/*.csv')
files.sort()
for file in files:
    regal = pd.concat([regal, pd.read_csv(file)], sort=False)
#%%
grant['Datetime'] = pd.to_datetime(grant['Datetime'])
grant = grant.sort_values('Datetime')

place_holder_times = grant.Datetime
del grant['Datetime']
grant = grant.astype(np.float64)
grant['Datetime'] = place_holder_times

grant.index = grant.Datetime
grant = grant.loc[start_time:end_time]
grant = grant.resample(interval).mean()
grant['Datetime'] = grant.index
grant['Location'] = 'Grant'


stevens['Datetime'] = pd.to_datetime(stevens['Datetime'])
stevens = stevens.sort_values('Datetime')

place_holder_times = stevens.Datetime
del stevens['Datetime']
stevens = stevens.astype(np.float64)
stevens['Datetime'] = place_holder_times

stevens.index = stevens.Datetime
stevens = stevens.loc[start_time:end_time] 
stevens = stevens.resample(interval).mean()
stevens['Datetime'] = stevens.index
stevens['Location'] = 'Stevens'


balboa['Datetime'] = pd.to_datetime(balboa['Datetime'])
balboa = balboa.sort_values('Datetime')

place_holder_times = balboa.Datetime
del balboa['Datetime']
balboa = balboa.astype(np.float64)
balboa['Datetime'] = place_holder_times

balboa.index = balboa.Datetime
balboa = balboa.loc[start_time:end_time] 
balboa = balboa.resample(interval).mean()
balboa['Datetime'] = balboa.index
balboa['Location'] = 'Balboa'


adams['Datetime'] = pd.to_datetime(adams['Datetime'])
adams = adams.sort_values('Datetime')

place_holder_times = adams.Datetime
del adams['Datetime']
adams = adams.astype(np.float64)
adams['Datetime'] = place_holder_times

adams.index = adams.Datetime
adams = adams.loc[start_time:end_time] 
adams = adams.resample(interval).mean()
adams['Datetime'] = adams.index
adams['Location'] = 'Adams'


jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson = jefferson.sort_values('Datetime')

place_holder_times = jefferson.Datetime
del jefferson['Datetime']
jefferson = jefferson.astype(np.float64)
jefferson['Datetime'] = place_holder_times

jefferson.index = jefferson.Datetime
jefferson = jefferson.loc[start_time:end_time] 
jefferson = jefferson.resample(interval).mean()
jefferson['Datetime'] = jefferson.index
jefferson['Location'] = 'Jefferson'


sheridan['Datetime'] = pd.to_datetime(sheridan['Datetime'])
sheridan = sheridan.sort_values('Datetime')

place_holder_times = sheridan.Datetime
del sheridan['Datetime']

sheridan = sheridan.astype(np.float64)
sheridan['Datetime'] = place_holder_times

sheridan.index = sheridan.Datetime
sheridan = sheridan.loc[start_time:end_time] 
sheridan = sheridan.resample(interval).mean()
sheridan['Datetime'] = sheridan.index
sheridan['Location'] = 'Sheridan'


browne['Datetime'] = pd.to_datetime(browne['Datetime'])
browne = browne.sort_values('Datetime')

place_holder_times = browne.Datetime
del browne['Datetime']
browne = browne.astype(np.float64)
browne['Datetime'] = place_holder_times

browne.index = browne.Datetime
browne = browne.loc[start_time:end_time] 
browne = browne.resample(interval).mean()
browne['Datetime'] = browne.index
browne['Location'] = 'Browne'


audubon['Datetime'] = pd.to_datetime(audubon['Datetime'])
audubon = audubon.sort_values('Datetime')

place_holder_times = audubon.Datetime
del audubon['Datetime']
audubon = audubon.astype(np.float64)
audubon['Datetime'] = place_holder_times

audubon.index = audubon.Datetime
audubon = audubon.loc[start_time:end_time] 
audubon = audubon.resample(interval).mean()
audubon['Datetime'] = audubon.index
audubon['Location'] = 'Audubon'


lidgerwood['Datetime'] = pd.to_datetime(lidgerwood['Datetime'])
lidgerwood = lidgerwood.sort_values('Datetime')

place_holder_times = lidgerwood.Datetime
del lidgerwood['Datetime']
lidgerwood = lidgerwood.astype(np.float64)
lidgerwood['Datetime'] = place_holder_times

lidgerwood.index = lidgerwood.Datetime
lidgerwood = lidgerwood.loc[start_time:end_time] 
lidgerwood = lidgerwood.resample(interval).mean()
lidgerwood['Datetime'] = lidgerwood.index
lidgerwood['Location'] = 'Lidgerwood'


regal['Datetime'] = pd.to_datetime(regal['Datetime'])
regal = regal.sort_values('Datetime')

place_holder_times = regal.Datetime
del regal['Datetime']
regal = regal.astype(np.float64)
regal['Datetime'] = place_holder_times

regal.index = regal.Datetime
regal = regal.loc[start_time:end_time] 
regal = regal.resample(interval).mean()
regal['Datetime'] = regal.index
regal['Location'] = 'Regal'
#%%

# Not needed for creating matched dataframes with reference measurements, just for looking at how indoor unit RH and temp compared to Clarity nodes RH and temp
# to see if it was feasible to use the Browne and Reference node Temp and RH across the indoor units to create MLR's for each. However, it does not look like
# the measurements are consistent enought across the indoor units to do that, and so instead will be just using basic linear regressions

audubon_bme = pd.DataFrame({})
audubon_bme_json = pd.DataFrame({})
audubon_bme, audubon_bme_json = load_indoor('Audubon', audubon_bme,audubon_bme_json, interval, start_time, end_time)
audubon_bme = audubon_bme.loc[start_time:end_time] 
#audubon_bme = audubon_bme.resample(interval).mean()
#audubon['temp'] = audubon_bme['temp']
#audubon['P'] = audubon_bme['P']
#audubon['RH'] = audubon_bme['RH']
#audubon = audubon.dropna()

adams_bme = pd.DataFrame({})
adams_bme_json = pd.DataFrame({})
adams_bme, adams_bme_json = load_indoor('Adams', adams_bme,adams_bme_json, interval, start_time, end_time)
adams_bme = adams_bme.loc[start_time:end_time] 
#adams_bme = adams_bme.resample(interval).mean()
#adams['temp'] = adams_bme['temp']
#adams['P'] = adams_bme['P']
#adams['RH'] = adams_bme['RH']
#adams = adams.dropna()

balboa_bme = pd.DataFrame({})
balboa_bme_json = pd.DataFrame({})
balboa_bme, balboa_bme_json = load_indoor('Balboa', balboa_bme,balboa_bme_json, interval, start_time, end_time)
balboa_bme = balboa_bme.loc[start_time:end_time] 
#balboa_bme = balboa_bme.resample(interval).mean()
#balboa['temp'] = balboa_bme['temp']
#balboa['P'] = balboa_bme['P']
#balboa['RH'] = balboa_bme['RH']
#balboa = balboa.dropna()

browne_bme = pd.DataFrame({})
browne_bme_json = pd.DataFrame({})
browne_bme, browne_bme_json = load_indoor('Browne', browne_bme,browne_bme_json, interval, start_time, end_time)
browne_bme = browne_bme.loc[start_time:end_time] 
#browne_bme = browne_bme.resample(interval).mean()
#browne['temp'] = browne_bme['temp']
#browne['P'] = browne_bme['P']
#browne['RH'] = browne_bme['RH']
#browne = browne.dropna()

grant_bme = pd.DataFrame({})
grant_bme_json = pd.DataFrame({})
grant_bme, grant_bme_json = load_indoor('Grant', grant_bme,grant_bme_json, interval, start_time, end_time)
grant_bme = grant_bme.loc[start_time:end_time] 
#grant_bme = grant_bme.resample(interval).mean()
#grant['temp'] = grant_bme['temp']
#grant['P'] = grant_bme['P']
#grant['RH'] = grant_bme['RH']
#grant = grant.dropna()

jefferson_bme = pd.DataFrame({})
jefferson_bme_json = pd.DataFrame({})
jefferson_bme, jefferson_bme_json = load_indoor('Jefferson', jefferson_bme,jefferson_bme_json, interval, start_time, end_time)
jefferson_bme = jefferson_bme.loc[start_time:end_time] 
#jefferson_bme = jefferson_bme.resample(interval).mean()
#jefferson['temp'] = jefferson_bme['temp']
#jefferson['P'] = jefferson_bme['P']
#jefferson['RH'] = jefferson_bme['RH']
#jefferson = jefferson.dropna()

lidgerwood_bme = pd.DataFrame({})
lidgerwood_bme_json = pd.DataFrame({})
lidgerwood_bme, lidgerwood_bme_json = load_indoor('Lidgerwood', lidgerwood_bme,lidgerwood_bme_json, interval, start_time, end_time)
lidgerwood_bme = lidgerwood_bme.loc[start_time:end_time] 
#lidgerwood_bme = lidgerwood_bme.resample(interval).mean()
#lidgerwood['temp'] = lidgerwood_bme['temp']
#lidgerwood['P'] = lidgerwood_bme['P']
#lidgerwood['RH'] = lidgerwood_bme['RH']
#lidgerwood = lidgerwood.dropna()

regal_bme = pd.DataFrame({})
regal_bme_json = pd.DataFrame({})
regal_bme, regal_bme_json = load_indoor('Regal', regal_bme,regal_bme_json, interval, start_time, end_time)
regal_bme = regal_bme.loc[start_time:end_time] 
#regal_bme = regal_bme.resample(interval).mean()
#regal['temp'] = regal_bme['temp']
#regal['P'] = regal_bme['P']
#regal['RH'] = regal_bme['RH']
#regal = regal.dropna()

sheridan_bme = pd.DataFrame({})
sheridan_bme_json = pd.DataFrame({})
sheridan_bme, sheridan_bme_json = load_indoor('Sheridan', sheridan_bme,sheridan_bme_json, interval, start_time, end_time)
sheridan_bme = sheridan_bme.loc[start_time:end_time] 
sheridan_bme = sheridan_bme.resample(interval).mean()
#sheridan['temp'] = sheridan_bme['temp']
#sheridan['P'] = sheridan_bme['P']
#sheridan['RH'] = sheridan_bme['RH']
#sheridan = sheridan.dropna()

stevens_bme = pd.DataFrame({})
stevens_bme_json = pd.DataFrame({})
stevens_bme, stevens_bme_json = load_indoor('Stevens', stevens_bme,stevens_bme_json, interval, start_time, end_time)
stevens_bme = stevens_bme.loc[start_time:end_time] 
#stevens_bme = stevens_bme.resample(interval).mean()
#stevens['temp'] = stevens_bme['temp']
#stevens['P'] = stevens_bme['P']
#stevens['RH'] = stevens_bme['RH']
#stevens = stevens.dropna()


audubon_bme = audubon_bme.dropna()
adams_bme = adams_bme.dropna()
balboa_bme = balboa_bme.dropna()
browne_bme = browne_bme.dropna()
grant_bme = grant_bme.dropna()
jefferson_bme = jefferson_bme.dropna()
lidgerwood_bme = lidgerwood_bme.dropna()
regal_bme = regal_bme.dropna()
sheridan_bme = sheridan_bme.dropna()
stevens_bme = stevens_bme.dropna()

#%%

Audubon_match_Ref = data_match(dt, Reference, audubon)    
#%%
Audubon_match_Browne = data_match(dt, Browne, audubon)
#%%
Adams_match_Ref = data_match(dt, Reference, adams)    
Adams_match_Browne = data_match(dt, Browne, adams)

Balboa_match_Ref = data_match(dt, Reference, balboa)    
Balboa_match_Browne = data_match(dt, Browne, balboa)

Browne_match_Ref = data_match(dt, Reference, browne)    
Browne_match_Browne = data_match(dt, Browne, browne)

Grant_match_Ref = data_match(dt, Reference, grant)    
Grant_match_Browne = data_match(dt, Browne, grant)

Jefferson_match_Ref = data_match(dt, Reference, jefferson)    
Jefferson_match_Browne = data_match(dt, Browne, jefferson)

Lidgerwood_match_Ref = data_match(dt, Reference, lidgerwood)    
Lidgerwood_match_Browne = data_match(dt, Browne, lidgerwood)

Regal_match_Ref = data_match(dt, Reference, regal)    
Regal_match_Browne = data_match(dt, Browne, regal)

Sheridan_match_Ref = data_match(dt, Reference, sheridan)    
Sheridan_match_Browne = data_match(dt, Browne, sheridan)

Stevens_match_Ref = data_match(dt, Reference, stevens)    
Stevens_match_Browne = data_match(dt, Browne, stevens)


    
#%%
# create df's with final raw calibration data to be used (indoor data matched with correct Clarity node data)
            
Audubon_final = final_df_cal_data(Audubon_match_Ref, Audubon_match_Browne)
Adams_final = final_df_cal_data(Adams_match_Ref, Adams_match_Browne)
Balboa_final = final_df_cal_data(Balboa_match_Ref, Balboa_match_Browne)
Browne_final = final_df_cal_data(Browne_match_Ref, Browne_match_Browne)
Grant_final = final_df_cal_data(Grant_match_Ref, Grant_match_Browne)
Jefferson_final = final_df_cal_data(Jefferson_match_Ref, Jefferson_match_Browne)
Lidgerwood_final = final_df_cal_data(Lidgerwood_match_Ref, Lidgerwood_match_Browne)
Regal_final = final_df_cal_data(Regal_match_Ref, Regal_match_Browne)
Sheridan_final = final_df_cal_data(Sheridan_match_Ref, Sheridan_match_Browne)
Stevens_final = final_df_cal_data(Stevens_match_Ref, Stevens_match_Browne)

#%%
Audubon_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Audubon_matched_df.csv', index=False)
Adams_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Adams_matched_df.csv', index=False)
Balboa_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Balboa_matched_df.csv', index=False)
Browne_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Browne_matched_df.csv', index=False)
Grant_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Grant_matched_df.csv', index=False)
Jefferson_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Jefferson_matched_df.csv', index=False)
Lidgerwood_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Lidgerwood_matched_df.csv', index=False)
Regal_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Regal_matched_df.csv', index=False)
Sheridan_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Sheridan_matched_df.csv', index=False)
Stevens_final.to_csv('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Stevens_matched_df.csv', index=False)

#%%
# add calibrated indoor data to df's (calibrated using power law fit)
Audubon_final = power_law_fit('Audubon', Audubon_final)
#%%
Adams_final = power_law_fit('Adams', Adams_final)
#%%
Balboa_final = power_law_fit('Balboa', Balboa_final)
#%%
Browne_final = power_law_fit('Browne', Browne_final)
#%%
Grant_final = power_law_fit('Grant', Grant_final)
#%%
Jefferson_final = power_law_fit('Jefferson', Jefferson_final)
#%%
Lidgerwood_final = power_law_fit('Lidgerwood', Lidgerwood_final)
#%%
Regal_final = power_law_fit('Regal', Regal_final)
#%%
Sheridan_final = power_law_fit('Sheridan', Sheridan_final)
#%%
Stevens_final = power_law_fit('Stevens', Stevens_final)

#%%
if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/clarity_PM2_5_time_series_legend_mute.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM 2.5 (ug/m3)')

p1.title.text = 'Clarity Calibrated PM 2.5'

p1.line(audubon.index,     audubon.PM2_5_env,     legend='Audubon',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
#p1.line(adams.index,       adams.PM2_5_env,       legend='Adams',         color='blue',        line_width=2, muted_color='blue', muted_alpha=0.2)
#p1.line(balboa.index,      balboa.PM2_5_env,      legend='Balboa',        color='red',         line_width=2, muted_color='red', muted_alpha=0.2)
#p1.line(browne.index,      browne.PM2_5_env,      legend='Browne',        color='black',       line_width=2, muted_color='black', muted_alpha=0.2)
#p1.line(grant.index,       grant.PM2_5_env,       legend='Grant',         color='purple',      line_width=2, muted_color='purple', muted_alpha=0.2)
#p1.line(jefferson.index,   jefferson.PM2_5_env,   legend='Jefferson',     color='brown',       line_width=2, muted_color='brown', muted_alpha=0.2)
#p1.line(lidgerwood.index,  lidgerwood.PM2_5_env,  legend='Lidgerwood',    color='orange',      line_width=2, muted_color='orange', muted_alpha=0.2)
#p1.line(regal.index,       regal.PM2_5_env,       legend='Regal',         color='khaki',       line_width=2, muted_color='khaki', muted_alpha=0.2)
#p1.line(sheridan.index,    sheridan.PM2_5_env,    legend='Sheridan',      color='deepskyblue', line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
#p1.line(stevens.index,     stevens.PM2_5_env,     legend='Stevens',       color='grey',        line_width=2, muted_color='grey', muted_alpha=0.2)
p1.line(Reference.index,   Reference.PM2_5_corrected,    legend='Reference',     color='olive',       line_width=2, muted_color='olive', muted_alpha=0.2)
p1.line(Browne.index,      Browne.PM2_5_corrected,    legend='Browne outside',     color='gold',       line_width=2, muted_color='gold', muted_alpha=0.2)


# For plotting matched data to ensure it looks good 
p1.line(Audubon_match_Ref.index,      Audubon_match_Ref.indoor_PM2_5_avg,    legend='Audubon averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Audubon_match_Ref.index,      Audubon_match_Ref.Reference_node_PM2_5_corrected,    legend='Reference Check func',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
p1.line(Audubon_match_Browne.index,      Audubon_match_Browne.indoor_PM2_5_avg,    legend='Audubon averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Audubon_match_Browne.index,      Audubon_match_Browne.Reference_node_PM2_5_corrected,    legend='Browne Check func',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Adams_match_Ref.index,      Adams_match_Ref.indoor_PM2_5_avg,    legend='Adams averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Adams_match_Browne.index,      Adams_match_Browne.indoor_PM2_5_avg,    legend='Adams averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Balboa_match_Ref.index,      Balboa_match_Ref.indoor_PM2_5_avg,    legend='Balboa averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Balboa_match_Browne.index,      Balboa_match_Browne.indoor_PM2_5_avg,    legend='Balboa averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Browne_match_Ref.index,      Browne_match_Ref.indoor_PM2_5_avg,    legend='Browne averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Browne_match_Browne.index,      Browne_match_Browne.indoor_PM2_5_avg,    legend='Browne averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Grant_match_Ref.index,      Grant_match_Ref.indoor_PM2_5_avg,    legend='Grant averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Grant_match_Browne.index,      Grant_match_Browne.indoor_PM2_5_avg,    legend='Grant averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Jefferson_match_Ref.index,      Jefferson_match_Ref.indoor_PM2_5_avg,    legend='Jefferson averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Jefferson_match_Browne.index,      Jefferson_match_Browne.indoor_PM2_5_avg,    legend='Jefferson averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Lidgerwood_match_Ref.index,      Lidgerwood_match_Ref.indoor_PM2_5_avg,    legend='Lidgerwood averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Lidgerwood_match_Browne.index,      Lidgerwood_match_Browne.indoor_PM2_5_avg,    legend='Lidgerwood averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Regal_match_Ref.index,      Regal_match_Ref.indoor_PM2_5_avg,    legend='Regal averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Regal_match_Browne.index,      Regal_match_Browne.indoor_PM2_5_avg,    legend='Regal averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Sheridan_match_Ref.index,      Sheridan_match_Ref.indoor_PM2_5_avg,    legend='Sheridan averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Sheridan_match_Browne.index,      Sheridan_match_Browne.indoor_PM2_5_avg,    legend='Sheridan averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Stevens_match_Ref.index,      Stevens_match_Ref.indoor_PM2_5_avg,    legend='Stevens averaged ref match',     color='purple',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Stevens_match_Browne.index,      Stevens_match_Browne.indoor_PM2_5_avg,    legend='Stevens averaged Browne match',     color='black',       line_width=2, muted_color='olive', muted_alpha=0.2)

# For plotting the final cal data set where the indoor data has been averaged and the data points selected based on the appropriate calibration regime
p1.line(Audubon_final.index,      Audubon_final.indoor,    legend='Audubon final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
p1.line(Audubon_final.index,      Audubon_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Adams_final.index,      Adams_final.indoor,    legend='Adams final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Adams_final.index,      Adams_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Balboa_final.index,      Balboa_final.indoor,    legend='Balboa final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Balboa_final.index,      Balboa_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Browne_final.index,      Browne_final.indoor,    legend='Browne final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Browne_final.index,      Browne_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Grant_final.index,      Grant_final.indoor,    legend='Grant final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Grant_final.index,      Grant_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Jefferson_final.index,      Jefferson_final.indoor,    legend='Jefferson final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Jefferson_final.index,      Jefferson_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Lidgerwood_final.index,      Lidgerwood_final.indoor,    legend='Lidgerwood final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Lidgerwood_final.index,      Lidgerwood_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Regal_final.index,      Regal_final.indoor,    legend='Regal final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Regal_final.index,      Regal_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Sheridan_final.index,      Sheridan_final.indoor,    legend='Sheridan final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Sheridan_final.index,      Sheridan_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Stevens_final.index,      Stevens_final.indoor,    legend='Stevens final',     color='orange',       line_width=2, muted_color='olive', muted_alpha=0.2)
#p1.line(Stevens_final.index,      Stevens_final.ref_value,    legend='Selected Ref Final',     color='lime',       line_width=2, muted_color='olive', muted_alpha=0.2)



p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Calibrated PM 2.5")

tabs = Tabs(tabs=[ tab1])

show(tabs)

#%%
if PlotType=='notebook':
    output_notebook()
else:
    output_file('/Users/matthew/Desktop/clarity_PM2_5_time_series_legend_mute.html')

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='RH (%)')

p1.title.text = 'RH'

p1.line(audubon_bme.index,     audubon_bme.RH,     legend='Audubon',       color='green',       line_width=2, muted_color='green', muted_alpha=0.2)
p1.line(adams_bme.index,       adams_bme.RH,       legend='Adams',         color='blue',        line_width=2, muted_color='blue', muted_alpha=0.2)
p1.line(balboa_bme.index,      balboa_bme.RH,      legend='Balboa',        color='teal',         line_width=2, muted_color='teal', muted_alpha=0.2)
p1.line(browne_bme.index,      browne_bme.RH,      legend='Browne',        color='gold',       line_width=2, muted_color='gold', muted_alpha=0.2)
p1.line(grant_bme.index,       grant_bme.RH,       legend='Grant',         color='purple',      line_width=2, muted_color='purple', muted_alpha=0.2)
p1.line(jefferson_bme.index,   jefferson_bme.RH,   legend='Jefferson',     color='brown',       line_width=2, muted_color='brown', muted_alpha=0.2)
p1.line(lidgerwood_bme.index,  lidgerwood_bme.RH,  legend='Lidgerwood',    color='orange',      line_width=2, muted_color='orange', muted_alpha=0.2)
p1.line(regal_bme.index,       regal_bme.RH,       legend='Regal',         color='khaki',       line_width=2, muted_color='khaki', muted_alpha=0.2)
p1.line(sheridan_bme.index,    sheridan_bme.RH,    legend='Sheridan',      color='deepskyblue', line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
p1.line(stevens_bme.index,     stevens_bme.RH,     legend='Stevens',       color='grey',        line_width=2, muted_color='grey', muted_alpha=0.2)
p1.line(Reference.index,   Reference.Rel_humid,    legend='Reference',     color='red',       line_width=2, muted_color='red', muted_alpha=0.2)
p1.line(Browne.index,      Browne.Rel_humid,    legend='Browne outside',     color='black',       line_width=2, muted_color='black', muted_alpha=0.2)


p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="RH")

tabs = Tabs(tabs=[ tab1])

show(tabs)
#%%
##############################
##############################

# Load in final calibration set csv's generated from above script in order to apply linear corrections to each region (and power law in middle)

grant = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Grant*.csv')
files.sort()
for file in files:
    grant = pd.concat([grant, pd.read_csv(file)], sort=False)


stevens = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Stevens*.csv')
files.sort()
for file in files:
    stevens = pd.concat([stevens, pd.read_csv(file)], sort=False)


balboa = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Balboa*.csv')
files.sort()
for file in files:
    balboa = pd.concat([balboa, pd.read_csv(file)], sort=False)


adams = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Adams*.csv')
files.sort()
for file in files:
    adams = pd.concat([adams, pd.read_csv(file)], sort=False)


jefferson = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Jefferson*.csv')
files.sort()
for file in files:
    jefferson = pd.concat([jefferson, pd.read_csv(file)], sort=False)


sheridan = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Sheridan*.csv')
files.sort()
for file in files:
    sheridan = pd.concat([sheridan, pd.read_csv(file)], sort=False)


# Comparison Data for indoor PMS5003 unit and Clarity Unit overlap for lowest 4 Clarity sensors

browne = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Browne*.csv')
files.sort()
for file in files:
    browne = pd.concat([browne, pd.read_csv(file)], sort=False)


audubon = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Audubon*.csv')
files.sort()
for file in files:
    audubon = pd.concat([audubon, pd.read_csv(file)], sort=False)


lidgerwood = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Lidgerwood*.csv')
files.sort()
for file in files:
    lidgerwood = pd.concat([lidgerwood, pd.read_csv(file)], sort=False)


regal = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/IndoorSensorCalibration_Nov2020/Regal*.csv')
files.sort()
for file in files:
    regal = pd.concat([regal, pd.read_csv(file)], sort=False)
#%%

adams['time'] = pd.to_datetime(adams['time'])
audubon['time'] = pd.to_datetime(audubon['time'])
balboa['time'] = pd.to_datetime(balboa['time'])
browne['time'] = pd.to_datetime(browne['time'])
grant['time'] = pd.to_datetime(grant['time'])
jefferson['time'] = pd.to_datetime(jefferson['time'])
lidgerwood['time'] = pd.to_datetime(lidgerwood['time'])
regal['time'] = pd.to_datetime(regal['time'])
sheridan['time'] = pd.to_datetime(sheridan['time'])
stevens['time'] = pd.to_datetime(stevens['time'])

adams = adams.sort_values('time')
audubon = audubon.sort_values('time')
balboa = balboa.sort_values('time')
browne = browne.sort_values('time')
grant = grant.sort_values('time')
jefferson = jefferson.sort_values('time')
lidgerwood = lidgerwood.sort_values('time')
regal = regal.sort_values('time')
sheridan = sheridan.sort_values('time')
stevens = stevens.sort_values('time')


adams.index = adams.time
audubon.index = audubon.time
balboa.index = balboa.time
browne.index = browne.time
grant.index = grant.time
jefferson.index = jefferson.time
lidgerwood.index = lidgerwood.time
regal.index = regal.time
sheridan.index = sheridan.time
stevens.index = stevens.time
#%%
# for checking for outliers in final cal df's so that can remove them (that is done below now after they are loaded in) - don't need these now (just for showing how got outlier values)
#Audubon_final = power_law_fit('Audubon', audubon)
#Adams_final = power_law_fit('Adams', adams)
#Balboa_final = power_law_fit('Balboa', balboa)
#Browne_final = power_law_fit('Browne', browne)
#Grant_final = power_law_fit('Grant', grant)
#Jefferson_final = power_law_fit('Jefferson', jefferson)
#Lidgerwood_final = power_law_fit('Lidgerwood', lidgerwood)
#Regal_final = power_law_fit('Regal', regal)
#Sheridan_final = power_law_fit('Sheridan', sheridan)
#Stevens_final = power_law_fit('Stevens', stevens)
#%%
# remove extreme outliers (determined by plotting final df's and looking at threshold of outliers)
audubon = audubon[audubon['indoor'] < 800]
adams = adams[adams['indoor'] < 800]

balboa = balboa[balboa['indoor'] < 1000]
balboa = balboa[balboa['indoor'].round(2) !=899.90]

browne = browne[browne['indoor'] < 1100]
browne = browne[browne['indoor'].round(2) !=988.22]

grant = grant[grant['indoor'] < 1000]
grant = grant[grant['indoor'].round(2) !=853.85]

jefferson = jefferson[jefferson['indoor'] < 1000]
jefferson = jefferson[jefferson['indoor'].round(2) !=759.28]

lidgerwood = lidgerwood[lidgerwood['indoor'] < 1100]
lidgerwood = lidgerwood[lidgerwood['indoor'].round(2) !=939.66]

regal = regal[regal['indoor'] < 1000]

sheridan = sheridan[sheridan['indoor'] < 1100]
sheridan = sheridan[sheridan['indoor'].round(2) !=819.49]

stevens = stevens[stevens['indoor'] < 1000]
stevens = stevens[stevens['indoor'].round(2) !=894.05]
#%%


p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Time (local)',
            y_axis_label='PM2.5 (ug/m^3)')

p1.title.text = 'Indoor PM2.5'

p1.line(audubon.index,     audubon.indoor,     legend='audubon',        color='green',             line_width=2, muted_color='green', muted_alpha=0.2)
p1.line(adams.index,       adams.indoor,       legend='adams',        color='blue',              line_width=2, muted_color='blue', muted_alpha=0.2)
p1.line(balboa.index,      balboa.indoor,      legend='balboa',        color='red',               line_width=2, muted_color='red', muted_alpha=0.2)
p1.line(browne.index,      browne.indoor,      legend='browne',        color='black',             line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(grant.index,       grant.indoor,       legend='grant',        color='purple',            line_width=2, muted_color='purple', muted_alpha=0.2)
p1.line(jefferson.index,   jefferson.indoor,   legend='jefferson',        color='brown',             line_width=2, muted_color='brown', muted_alpha=0.2)
p1.line(lidgerwood.index,  lidgerwood.indoor,  legend='lidgerwood',        color='orange',            line_width=2, muted_color='orange', muted_alpha=0.2)
p1.line(regal.index,       regal.indoor,       legend='regal',        color='khaki',             line_width=2, muted_color='khaki', muted_alpha=0.2)
p1.line(sheridan.index,    sheridan.indoor,    legend='sheridan',        color='deepskyblue',       line_width=2, muted_color='deepskyblue', muted_alpha=0.2)
p1.line(stevens.index,     stevens.indoor,     legend='stevens',       color='grey',              line_width=2, muted_color='grey', muted_alpha=0.2)

# all ref values are the same between units
p1.line(audubon.index,     audubon.ref_value,     legend='ref_value',        color='teal',             line_width=2, muted_color='teal', muted_alpha=0.2)



p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Indoor PM 2.5")

tabs = Tabs(tabs=[ tab1])

show(tabs)    
#%%

audubon_low = audubon[audubon['ref_value'] < 68]
audubon_high = audubon[audubon['ref_value'] > 68]

adams_low = adams[adams['ref_value'] < 68]
adams_high = adams[adams['ref_value'] > 68]

balboa_low = balboa[balboa['ref_value'] < 68]
balboa_high = balboa[balboa['ref_value'] > 68]

browne_low = browne[browne['ref_value'] < 68]
browne_high = browne[browne['ref_value'] > 68]

grant_low = grant[grant['ref_value'] < 68]
grant_high = grant[grant['ref_value'] > 68]

jefferson_low = jefferson[jefferson['ref_value'] < 68]
jefferson_high = jefferson[jefferson['ref_value'] > 68]

lidgerwood_low = lidgerwood[lidgerwood['ref_value'] < 68]
lidgerwood_high = lidgerwood[lidgerwood['ref_value'] > 68]

regal_low = regal[regal['ref_value'] < 68]
regal_high = regal[regal['ref_value'] > 68]

sheridan_low = sheridan[sheridan['ref_value'] < 68]
sheridan_high = sheridan[sheridan['ref_value'] > 68]

stevens_low = stevens[stevens['ref_value'] < 68]
stevens_high = stevens[stevens['ref_value'] > 68]

audubon['indoor_corrected'] = np.where(audubon.indoor > 68, (audubon.indoor-0.6)*(1/0.66), 
         (np.where(audubon.indoor < 68, (audubon.indoor-1.49)/2.04, audubon.indoor)))

adams['indoor_corrected'] = np.where(adams.indoor > 68, (adams.indoor-9)*(1/0.63), 
         (np.where(adams.indoor < 68, (adams.indoor-0.97)/1.9, adams.indoor)))

balboa['indoor_corrected'] = np.where(balboa.indoor > 68, (balboa.indoor+18.16)*(1/0.83), 
         (np.where(balboa.indoor < 68, (balboa.indoor-1.25)/2.02, balboa.indoor)))

browne['indoor_corrected'] = np.where(browne.indoor > 68, (browne.indoor+27.43)*(1/0.94), 
         (np.where(browne.indoor < 68, (browne.indoor-0.36)/2.09, browne.indoor)))

grant['indoor_corrected'] = np.where(grant.indoor > 68, (grant.indoor+7.68)*(1/0.85), 
         (np.where(grant.indoor < 68, (grant.indoor-0.88)/2.12, grant.indoor)))

jefferson['indoor_corrected'] = np.where(jefferson.indoor > 68, (jefferson.indoor-2.17)*(1/0.71), 
         (np.where(jefferson.indoor < 68, (jefferson.indoor-0.78)/1.84, jefferson.indoor)))

lidgerwood['indoor_corrected'] = np.where(lidgerwood.indoor > 68, (lidgerwood.indoor+16.05)*(1/0.91), 
         (np.where(lidgerwood.indoor < 68, (lidgerwood.indoor-1.08)/2.11, lidgerwood.indoor)))

regal['indoor_corrected'] = np.where(regal.indoor > 68, (regal.indoor-4.12)*(1/0.72), 
         (np.where(regal.indoor < 68, (regal.indoor-1.14)/1.99, regal.indoor)))

sheridan['indoor_corrected'] = np.where(sheridan.indoor > 68, (sheridan.indoor-1.26)*(1/0.62), 
         (np.where(sheridan.indoor < 68, (sheridan.indoor-1.16)/2.07, sheridan.indoor)))

stevens['indoor_corrected'] = np.where(stevens.indoor > 68, (stevens.indoor-9.03)*(1/0.67), 
         (np.where(stevens.indoor < 68, (stevens.indoor-0.62)/2.01, stevens.indoor)))
#%%
audubon['prediction_residuals'] = audubon['ref_value'] - audubon['indoor_corrected']
adams['prediction_residuals'] = adams['ref_value'] - adams['indoor_corrected']
balboa['prediction_residuals'] = balboa['ref_value'] - balboa['indoor_corrected']
browne['prediction_residuals'] = browne['ref_value'] - browne['indoor_corrected']
grant['prediction_residuals'] = grant['ref_value'] - grant['indoor_corrected']
jefferson['prediction_residuals'] = jefferson['ref_value'] - jefferson['indoor_corrected']
lidgerwood['prediction_residuals'] = lidgerwood['ref_value'] - lidgerwood['indoor_corrected']
regal['prediction_residuals'] = regal['ref_value'] - regal['indoor_corrected']
sheridan['prediction_residuals'] = sheridan['ref_value'] - sheridan['indoor_corrected']
stevens['prediction_residuals'] = stevens['ref_value'] - stevens['indoor_corrected']
#%%
# just for checking performance on for data pairs where the reference data is > 35 ug/m3
audubon = audubon[audubon['ref_value'] > 35]
adams = adams[adams['ref_value'] > 35]
balboa = balboa[balboa['ref_value'] > 35]
browne = browne[browne['ref_value'] > 35]
grant = grant[grant['ref_value'] > 35]
jefferson = jefferson[jefferson['ref_value'] > 35]
lidgerwood = lidgerwood[lidgerwood['ref_value'] > 35]
regal = regal[regal['ref_value'] > 35]
sheridan = sheridan[sheridan['ref_value'] > 35]
stevens = stevens[stevens['ref_value'] > 35]

#%%

# if calculating the performance of the calibrated data, use #lines = 1 and residuals check =1, and add in residuals = indoor.prediction_residuals

#linear_plot(audubon_low.ref_value, audubon_low.indoor, audubon_high.ref_value, audubon_high.indoor,'audubon', 2)
linear_plot(audubon.ref_value, audubon.indoor_corrected, audubon.ref_value, audubon.indoor,'audubon', 1, residuals_check = 1, residuals = audubon.prediction_residuals)
#%%
#linear_plot(adams_low.ref_value, adams_low.indoor, adams_high.ref_value, adams_high.indoor,'adams', 2)
linear_plot(adams.ref_value, adams.indoor_corrected, adams.ref_value, adams.indoor,'Adams', 1, residuals_check = 1, residuals = adams.prediction_residuals)
#%%
#linear_plot(balboa_low.ref_value, balboa_low.indoor, balboa_high.ref_value, balboa_high.indoor,'Balboa', 2)
linear_plot(balboa.ref_value, balboa.indoor_corrected, balboa.ref_value, balboa.indoor,'Balboa', 1, residuals_check = 1, residuals = balboa.prediction_residuals)
#%%
#linear_plot(browne_low.ref_value, browne_low.indoor, browne_high.ref_value, browne_high.indoor,'Browne', 2)
linear_plot(browne.ref_value, browne.indoor_corrected, browne.ref_value, browne.indoor,'Browne', 1, residuals_check = 1, residuals = browne.prediction_residuals)
#%%
#linear_plot(grant_low.ref_value, grant_low.indoor, grant_high.ref_value, grant_high.indoor,'Grant', 2)
linear_plot(grant.ref_value, grant.indoor_corrected, grant.ref_value, grant.indoor,'Grant', 1, residuals_check = 1, residuals = grant.prediction_residuals)
#%%
#linear_plot(jefferson_low.ref_value, jefferson_low.indoor, jefferson_high.ref_value, jefferson_high.indoor,'Jefferson', 2)
linear_plot(jefferson.ref_value, jefferson.indoor_corrected, jefferson.ref_value, jefferson.indoor,'Jefferson', 1, residuals_check = 1, residuals = jefferson.prediction_residuals)
#%%
#linear_plot(lidgerwood_low.ref_value, lidgerwood_low.indoor, lidgerwood_high.ref_value, lidgerwood_high.indoor,'Lidgerwood', 2)
linear_plot(lidgerwood.ref_value, lidgerwood.indoor_corrected, lidgerwood.ref_value, lidgerwood.indoor,'Lidgerwood', 1, residuals_check = 1, residuals = lidgerwood.prediction_residuals)
#%%
#linear_plot(regal_low.ref_value, regal_low.indoor, regal_high.ref_value, regal_high.indoor,'Regal', 2)
linear_plot(regal.ref_value, regal.indoor_corrected, regal.ref_value, regal.indoor,'Regal', 1, residuals_check = 1, residuals = regal.prediction_residuals)
#%%
#linear_plot(sheridan_low.ref_value, sheridan_low.indoor, sheridan_high.ref_value, sheridan_high.indoor,'Sheridan', 2)
linear_plot(sheridan.ref_value, sheridan.indoor_corrected, sheridan.ref_value, sheridan.indoor,'Sheridan', 1, residuals_check = 1, residuals = sheridan.prediction_residuals)
#%%
#linear_plot(stevens_low.ref_value, stevens_low.indoor, stevens_high.ref_value, stevens_high.indoor,'Stevens', 2)
linear_plot(stevens.ref_value, stevens.indoor_corrected, stevens.ref_value, stevens.indoor,'Stevens', 1, residuals_check = 1, residuals = stevens.prediction_residuals)






























