#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 17:31:49 2021

@author: matthew
"""

import pandas as pd
from glob import glob
import matplotlib as plt
import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
from bokeh.io import export_png
from bokeh.layouts import gridplot
import metpy
from scipy import stats
from scipy import optimize  
import statsmodels.api as sm
import statistics
from gaussian_fit_function import gaussian_fit


from limit_of_detection import lod
import copy
from random_forest_function_test import rf, evaluate_model

from Augusta_hybrid_calibration import hybrid_function
from linear_plot_function import linear_plot
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate, cross_val_score
from sklearn.datasets import load_iris, load_boston
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
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


# Use BAM data from SRCAA
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)


#%%
#threshold = 3.48         # threshold for splitting df's for "blank" measurements of BAM for calculating LOD

# Choose dates of interest
# Augusta overlap period
start_time = '2019-12-17 15:00'
end_time = '2020-03-05 23:00'

# choose number of kfolds for cv
number_folds = 5
# for straight kfold cv
kf = KFold(n_splits=number_folds, shuffle=False)

# for stratified cv
#kf = StratifiedKFold(n_splits=number_folds, shuffle=False)
#%%
# choose resample interval
#interval = '60T'
interval = '24H'

#%%

def kfold_eval(interval, correction_type, kf, Augusta_data, Clarity_data, start_time, end_time):

    Clarity_data = Clarity_data.copy()
    Augusta_data = Augusta_data.copy()
    
    
    Clarity_data['time'] = pd.to_datetime(Clarity_data['time'])
    Clarity_data = Clarity_data.sort_values('time')
    Clarity_data.index = Clarity_data.time
    Clarity = Clarity_data.loc[start_time:end_time]
    Clarity = Clarity.resample(interval).mean()  
    
    Augusta_data['time'] = pd.to_datetime(Augusta_data['time'])
    Augusta_data = Augusta_data.sort_values('time')
    Augusta_data.index = Augusta_data.time
    Augusta = Augusta_data.loc[start_time:end_time]
    Augusta = Augusta.resample(interval).mean()
    Augusta['time'] = Augusta.index
   
    if interval == '60T':
        #drop last row so same number of measurements as Paccar and Reference Nodes
        Augusta = Augusta[:-1]
    else:
        pass
    
    features = ['PM2_5', 'Rel_humid', 'temp']

    combined_df = pd.DataFrame({})
    combined_df['PM2_5'] = Clarity['PM2_5']
    combined_df['Rel_humid'] = Clarity['Rel_humid']
    combined_df['temp'] = Clarity['temp']

    combined_df['target'] = Augusta['PM2_5']
    
 #   return Clarity, Augusta, combined_df

    
    i = 1

    eval_df = pd.DataFrame({})
    MAE = []
    r_squared = []
    MSE = []
    MBE = []

    if correction_type == 'mlr':

        for train_index, test_index in kf.split(combined_df):
            X_train = combined_df.iloc[train_index].loc[:, features]
            X_train = sm.add_constant(X_train) ## let's add an intercept (beta_0) to our model
            #  print(X_train)
            X_test = combined_df.iloc[test_index][features]
            X_test = sm.add_constant(X_test) ## let's add an intercept (beta_0) to our model
            #  print(X_test)
            y_train = combined_df.iloc[train_index].loc[:,'target']
            #  print(y_train)
            y_test = combined_df.iloc[test_index]['target']
            #  print(y_test)
            #Train the model
            #model.fit(X_train, y_train) #Training the model
            mlr_model = sm.OLS(y_train, X_train).fit() #Training the model
            print_model = mlr_model.summary()
            print(print_model)
            
            # print(len(mlr_model.predict(X_test)))
            MBE_loop = (((mlr_model.predict(X_test) - y_test)).sum())/len(y_test)
            print(MBE_loop)
            
            #r2 = r2_score(y_test, mlr_model.predict(X_test))
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y_test, mlr_model.predict(X_test))
            r2 = r_value**2
            print('r^2 = ', r2)
            print('intercept = ', intercept)
            MAE_loop = mean_absolute_error(y_test, mlr_model.predict(X_test))
            MSE_loop = mean_squared_error(y_test, mlr_model.predict(X_test))
          #  print(f"R^2 for the fold no. {i} on the test set: {r2_score(y_test, mlr_model.predict(X_test))}")
            print(f"MAE for the fold no. {i} on the test set: {mean_absolute_error(y_test, mlr_model.predict(X_test))}")
            # print(MAE_loop)
            print(f"Mean Squared Error for the fold no. {i} on the test set: {mean_squared_error(y_test, mlr_model.predict(X_test))}")
            # print(MSE_loop)
            print()
            MAE.append(MAE_loop)
            r_squared.append(r2)
            MSE.append(MSE_loop)
            MBE.append(MBE_loop)
            i += 1
            print(i)
    
        eval_df['MAE'] = MAE
        eval_df['r_squared'] = r_squared
        eval_df['MSE'] = MSE 
        eval_df['RMSE'] = eval_df['MSE']**(1/2)   
        eval_df['MBE'] = MBE
     

        #just checking outputs for manual calcs
       # check_df = ({})
       # if i == 9:
       #     check_df['y_test'] = y_test
       #     check_df['X_test'] = mlr_model.predict(X_test)
       # boxplot = eval_df.boxplot(column=['r_squared', 'MAE', 'RMSE', 'MBE'])   
        # boxplot = eval_df.boxplot(column=['MAE'])  
    
    elif correction_type == 'raw':
        
        X_test = combined_df['PM2_5']
        X_test = sm.add_constant(X_test) ## let's add an intercept (beta_0) to our model
        #  print(X_test)
        y_test = combined_df['target']

        mlr_model = sm.OLS(y_test, X_test).fit() #Training the model - but use all available raw data
 
        
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(combined_df['target'], combined_df['PM2_5'])
        r2 = r_value**2
        
       # print('slope = ', slope)
       # print('intercept = ', intercept)
        print('r^2 = ', r2)
        print('intercept = ', intercept)
        print('slope = ', slope)
        
        MBE_loop = ((combined_df['PM2_5'] - combined_df['target'])).sum()/len(combined_df['PM2_5'])
        print('MBE= ', MBE_loop)
        MAE_loop = mean_absolute_error(combined_df['target'],combined_df['PM2_5'])
        MSE_loop = mean_squared_error(combined_df['target'],combined_df['PM2_5'])
        print(f"MAE for the fold no. {i} on the test set: {mean_absolute_error(combined_df['target'],combined_df['PM2_5'])}")
        # print(MAE_loop)
        print(f"Mean Squared Error for the fold no. {i} on the test set: {mean_squared_error(combined_df['target'],combined_df['PM2_5'])}")
        # print(MSE_loop)
        print()
        MAE.append(MAE_loop)
        r_squared.append(r2)
        MSE.append(MSE_loop)
        MBE.append(MBE_loop)
        i += 1
    
        eval_df['MAE'] = MAE
        eval_df['r_squared'] = r_squared
        eval_df['MSE'] = MSE 
        eval_df['RMSE'] = eval_df['MSE']**(1/2)   
        eval_df['MBE'] = MBE
    
        stdev_resid = (combined_df['PM2_5'] - combined_df['target']).std()
        print('stdev residuals = ', stdev_resid)
        
    
     #   boxplot = eval_df.boxplot(column=['r_squared', 'MAE', 'RMSE', 'MBE'])   
        
    else:
        pass

    
    #return eval_df # just for checking manual calcs:, boxplot, combined_df, mlr_model.predict(X_test), y_test, check_df
    return eval_df  # just for checking manual calcs:, boxplot, combined_df, mlr_model.predict(X_test), y_test, check_df, check_df



#%%
### MAKE SURE TO CHANGE INTERVAL TO 60T before making these

paccar_eval_df_hourly_mlr = kfold_eval(interval, 'mlr', kf, Augusta_All, Paccar_All, start_time, end_time)
#df_check = pd.DataFrame(check_df)
#%%
paccar_eval_df_hourly_raw = kfold_eval(interval, 'raw', kf, Augusta_All, Paccar_All, start_time, end_time)
#%%
#X_test, y_test = kfold_eval(interval, 'raw', kf, Augusta_All, Paccar_All, start_time, end_time)

reference_eval_df_hourly_mlr = kfold_eval(interval, 'mlr', kf, Augusta_All, Reference_All, start_time, end_time)
#%%
reference_eval_df_hourly_raw = kfold_eval(interval, 'raw', kf, Augusta_All, Reference_All, start_time, end_time)

#%%
### MAKE SURE TO CHANGE INTERVAL TO 24hr before making these

paccar_eval_df_daily_mlr = kfold_eval(interval, 'mlr', kf, Augusta_All, Paccar_All, start_time, end_time)
#%%
paccar_eval_df_daily_raw = kfold_eval(interval, 'raw', kf, Augusta_All, Paccar_All, start_time, end_time)
#%%
reference_eval_df_daily_mlr = kfold_eval(interval, 'mlr', kf, Augusta_All, Reference_All, start_time, end_time)
#%%
#df_check = pd.DataFrame(check_df)
reference_eval_df_daily_raw = kfold_eval(interval, 'raw', kf, Augusta_All, Reference_All, start_time, end_time)
#%%
# Create df for each plot so can plot directly using boxplot command

hourly_MBE = pd.DataFrame({})
hourly_MBE['Unit 1 MLR'] = paccar_eval_df_hourly_mlr['MBE']
hourly_MBE['Unit 2 MLR'] = reference_eval_df_hourly_mlr['MBE']
hourly_MBE['Unit 1 Raw'] = paccar_eval_df_hourly_raw['MBE']
hourly_MBE['Unit 2 Raw'] = reference_eval_df_hourly_raw['MBE']

#hourly_MBE_list = hourly_MBE.transpose().values.tolist()

hourly_MAE = pd.DataFrame({})
hourly_MAE['Unit 1 MLR'] = paccar_eval_df_hourly_mlr['MAE']
hourly_MAE['Unit 2 MLR'] = reference_eval_df_hourly_mlr['MAE']
hourly_MAE['Unit 1 Raw'] = paccar_eval_df_hourly_raw['MAE']
hourly_MAE['Unit 2 Raw'] = reference_eval_df_hourly_raw['MAE']

#hourly_MAE_list = hourly_MAE.transpose().values.tolist()

daily_MBE = pd.DataFrame({})
daily_MBE['Unit 1 MLR'] = paccar_eval_df_daily_mlr['MBE']
daily_MBE['Unit 2 MLR'] = reference_eval_df_daily_mlr['MBE']
daily_MBE['Unit 1 Raw'] = paccar_eval_df_daily_raw['MBE']
daily_MBE['Unit 2 Raw'] = reference_eval_df_daily_raw['MBE']

#daily_MBE_list = daily_MBE.transpose().values.tolist()

daily_MAE = pd.DataFrame({})
daily_MAE['Unit 1 MLR'] = paccar_eval_df_daily_mlr['MAE']
daily_MAE['Unit 2 MLR'] = reference_eval_df_daily_mlr['MAE']
daily_MAE['Unit 1 Raw'] = paccar_eval_df_daily_raw['MAE']
daily_MAE['Unit 2 Raw'] = reference_eval_df_daily_raw['MAE']

#daily_MAE_list = daily_MAE.transpose().values.tolist()

hourly_RMSE = pd.DataFrame({})
hourly_RMSE['Unit 1 MLR'] = paccar_eval_df_hourly_mlr['RMSE']
hourly_RMSE['Unit 2 MLR'] = reference_eval_df_hourly_mlr['RMSE']
hourly_RMSE['Unit 1 Raw'] = paccar_eval_df_hourly_raw['RMSE']
hourly_RMSE['Unit 2 Raw'] = reference_eval_df_hourly_raw['RMSE']

hourly_r_squared = pd.DataFrame({})
hourly_r_squared['Unit 1 MLR'] = paccar_eval_df_hourly_mlr['r_squared']
hourly_r_squared['Unit 2 MLR'] = reference_eval_df_hourly_mlr['r_squared']
hourly_r_squared['Unit 1 Raw'] = paccar_eval_df_hourly_raw['r_squared']
hourly_r_squared['Unit 2 Raw'] = reference_eval_df_hourly_raw['r_squared']

daily_RMSE = pd.DataFrame({})
daily_RMSE['Unit 1 MLR'] = paccar_eval_df_daily_mlr['RMSE']
daily_RMSE['Unit 2 MLR'] = reference_eval_df_daily_mlr['RMSE']
daily_RMSE['Unit 1 Raw'] = paccar_eval_df_daily_raw['RMSE']
daily_RMSE['Unit 2 Raw'] = reference_eval_df_daily_raw['RMSE']

#daily_MBE_list = daily_MBE.transpose().values.tolist()

daily_r_squared = pd.DataFrame({})
daily_r_squared['Unit 1 MLR'] = paccar_eval_df_daily_mlr['r_squared']
daily_r_squared['Unit 2 MLR'] = reference_eval_df_daily_mlr['r_squared']
daily_r_squared['Unit 1 Raw'] = paccar_eval_df_daily_raw['r_squared']
daily_r_squared['Unit 2 Raw'] = reference_eval_df_daily_raw['r_squared']


#%%
# create boxplots for gridplot
#boxplot_1 = hourly_MBE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR'])   
#boxplot_2 = hourly_MAE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR'])   
#boxplot_3 = daily_MBE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR'])   
#boxplot_4 = daily_MAE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR'])   

#%%

# create boxplot gridplot of cv evaluations for the Clarity Nodes collocated with Augusta BAM
plt.rcParams["font.family"] = "Times New Roman"
plt.rc("figure", facecolor="white")

fig, axs = plt.subplots(4, 2, figsize=(8,10))

# boxplot of hourly resample MBE
plt.axes(axs[0,0])
hourly_MBE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR'])   
axs[0, 0].set_title('Hourly Resample', fontsize = 16)
axs[0, 0].set_ylim([-4, 8])
axs[0, 0].tick_params(axis="y", labelsize=12)
axs[0, 0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
axs[0, 0].set_xticklabels([])

# boxplot of hourly resample MAE
plt.axes(axs[1,0])
hourly_MAE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR']) 
axs[1, 0].set_title('')
axs[1, 0].set_ylim([-1, 7])
axs[1, 0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
axs[1, 0].tick_params(axis="y", labelsize=12)
axs[1, 0].set_xticklabels([])


# boxplot of hourly resample RMSE
plt.axes(axs[2,0])
hourly_RMSE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR'])   
#axs[0, 0].set_title('Hourly Resample', fontsize = 16)
axs[2, 0].set_ylim([0, 10])
axs[2, 0].tick_params(axis="y", labelsize=12)
axs[2, 0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
axs[2, 0].set_xticklabels([])

# boxplot of hourly resample r^2
plt.axes(axs[3,0])
hourly_r_squared.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR']) 
axs[3, 0].set_title('')
#axs[3, 0].set_ylim([-1, 7])
axs[3, 0].set_ylim([0.6, 1])
axs[3, 0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
plt.xticks(rotation=315, ha='left', fontsize = 14)
axs[3, 0].tick_params(axis="y", labelsize=12)

# boxplot of daily resample MBE
plt.axes(axs[0,1])
daily_MBE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR']) 
axs[0, 1].set_title('24 Hr Resample', fontsize = 16)
axs[0, 1].set_ylim([-4, 8])
axs[0, 1].yaxis.set_label_position("right")
axs[0, 1].set_ylabel('MBE', rotation = 270, fontsize = 14)
axs[0, 1].yaxis.set_label_coords(1.075,0.5)
axs[0, 1].set_xticklabels([])
axs[0, 1].tick_params(axis="y", labelsize=12)

# boxplot of daily resample MAE
plt.axes(axs[1,1])
daily_MAE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR']) 
axs[1, 1].set_title("")
axs[1, 1].set_ylim([-1, 7])
axs[1, 1].yaxis.set_label_position("right")
axs[1, 1].set_ylabel('MAE', rotation = 270, fontsize = 14)
axs[1, 1].yaxis.set_label_coords(1.075,0.5)
axs[1, 1].tick_params(axis="y", labelsize=12)
axs[1, 1].set_xticklabels([])

# boxplot of daily resample RMSE
plt.axes(axs[2,1])
daily_RMSE.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR']) 
#axs[2, 1].set_ylim([-4, 8])
axs[2, 1].yaxis.set_label_position("right")
axs[2, 1].set_ylabel('RMSE', rotation = 270, fontsize = 14)
axs[2, 1].yaxis.set_label_coords(1.075,0.5)
axs[2, 1].set_ylim([0, 10])
axs[2, 1].set_xticklabels([])
axs[2, 1].tick_params(axis="y", labelsize=12)

# boxplot of daily resample r^2
plt.axes(axs[3,1])
daily_r_squared.boxplot(column=['Unit 1 Raw', 'Unit 2 Raw', 'Unit 1 MLR', 'Unit 2 MLR']) 
axs[3, 1].set_title("")
#axs[3, 1].set_ylim([-1, 7])
axs[3, 1].yaxis.set_label_position("right")
axs[3, 1].set_ylabel('R²', rotation = 270, fontsize = 14)
axs[3, 1].yaxis.set_label_coords(1.075,0.5)
axs[3, 1].set_ylim([0.6, 1])
axs[3, 1].tick_params(axis="y", labelsize=12)
plt.xticks(rotation=315, ha='left', fontsize = 14)


plt.tight_layout()

plt.savefig('/Users/matthew/Desktop/thesis/publishing/kfold_eval.png')

plt.show()


#%%

# function to resample all data to timeframe and interval needed

def resample(data, interval, start_time, stop_time, location):
    
    data = data.copy()

    data['time'] = pd.to_datetime(data['time'])
    data = data.sort_values('time')
    data.index = data.time
    data = data.loc[start_time:end_time]
    data = data.resample(interval).mean()  

    if location == 'Augusta_All':
        if interval == '60T':
           #drop last row so same number of measurements as Paccar and Reference Nodes
            data = data[:-1]
        else:
            pass
    
    return data

# function for plotting raw and corrected time series

def time_series(Clarity1, Clarity2, Augusta, name, resample):
   
    fig = plt.subplot()#figsize=(12,5))
   # plt.axes(axs)
    
    if resample == 'raw':
    
        ax1 = Clarity1.PM2_5.plot(color='green', grid=True, label='Unit 1')
        ax2 = Clarity2.PM2_5.plot(color='black', grid=True, label='Unit 2')
        ax3 = Augusta.PM2_5.plot(color='red', grid=True, label = 'BAM')#, secondary_y=True, label='Sum')
    
        plt.legend(loc='upper right')
        ax1.set(xlabel="", ylabel="PM 2.5 (ug/m³)")
    
    elif resample == 'resample':
        
        ax1 =Clarity1.PM2_5_corrected.plot(color='green', grid=True, label='Unit 1')
        ax2 = Clarity2.PM2_5_corrected.plot(color='black', grid=True, label='Unit 2')
        ax3 = Augusta.PM2_5.plot(color='red', grid=True, label = 'BAM')#, secondary_y=True, label='Sum')
    
        plt.legend(loc='upper right')
        ax1.set(xlabel="", ylabel="PM 2.5 (ug/m³)")

    plt.savefig('/Users/matthew/Desktop/thesis/publishing/time_series_' + name + '.png')
    plt.show()
    
    
    
    return fig

# function for using whole dataset for mlr correction
    
def mlr_function(Clarity, Augusta):

    X = Clarity[['PM2_5','Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)  Rel_humid
  #  X = X.dropna()
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    y = Augusta['PM2_5']
    print(len(X))
    print(len(y))
    mlr_model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    # Note the difference in argument order
    predictions = mlr_model.predict(X)
    
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(y, predictions)
    r2 = r_value**2

    print('r^2 = ', r2)
    print('intercept = ', intercept)
    print('slope = ', slope)
    
    mae = (abs(predictions - y).sum())/len(y)
    stdev_resid = (predictions - y).std()
    
    # Print out the statistics
    print_model = mlr_model.summary()
    print(print_model)
    Clarity['PM2_5_corrected'] = predictions
    
    print('MAE = ', mae)
    print('Calibrated Residual Stdev = ', stdev_resid)
    
    return Clarity

#%%
# choose resample interval
#interval = '60T'
interval = '24H'
#%%
# resample data
# hourly resample
Paccar_hourly = resample(Paccar_All, interval, start_time, end_time, 'Paccar_All')
Reference_hourly = resample(Reference_All, interval, start_time, end_time, 'Refrence_All')
Augusta_hourly = resample(Augusta_All, interval, start_time, end_time, 'Augusta_All')
#%%
Paccar_hourly = mlr_function(Paccar_hourly, Augusta_hourly) 
#%%
Reference_hourly = mlr_function(Reference_hourly, Augusta_hourly)

#%%
# daily resample
Paccar_daily = resample(Paccar_All, interval, start_time, end_time, 'Paccar_All')
#%%
Reference_daily = resample(Reference_All, interval, start_time, end_time, 'Reference_All')
#%%
Augusta_daily = resample(Augusta_All, interval, start_time, end_time, 'Augusta_All')
#%%
Paccar_daily = mlr_function(Paccar_daily, Augusta_daily)
#%%
Reference_daily = mlr_function(Reference_daily, Augusta_daily)
#%%
Reference_daily['empty'] = np.nan

#%%

# plot time series and scatters of raw, resampled (hourly and 24 hour data)

fig_1 = time_series(Reference_hourly, Paccar_hourly, Augusta_hourly,'hourly_raw', 'raw')
fig_2 = time_series(Reference_daily, Paccar_daily, Augusta_daily,'daily_raw', 'raw')
fig_3 = time_series(Reference_hourly, Paccar_hourly, Augusta_hourly,'hourly_resample_corrected', 'resample')
fig_4 = time_series(Reference_daily, Paccar_daily, Augusta_daily,'daily_resample_corrected', 'resample')

fig_list = [fig_1, fig_2, fig_3, fig_4]
#%%
# combine time series plots into gridplot
import matplotlib.dates

plt.rcParams['savefig.facecolor']='white'
#fig, axs = plt.subplots(2, 2, figsize=(8,8), facecolor='white')
fig, axs = plt.subplots(2, 1, figsize=(8,6), facecolor='white')


axs[0].set_title('Raw Measurements', fontsize = 16)
axs[0].plot(Paccar_hourly.index, Paccar_hourly.PM2_5)
axs[0].plot(Reference_hourly.index, Reference_hourly.PM2_5)
axs[0].plot(Augusta_hourly.index, Augusta_hourly.PM2_5)
axs[0].set_ylim([-5, 70])
#axs[0].yaxis.set_label_position("right")
#axs[0].set_ylabel('Raw Measurements', rotation = 270, fontsize = 14)
#axs[0].yaxis.set_label_coords(1.025,0.5)
axs[0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
axs[0].tick_params(axis="y", labelsize=12)
axs[0].set_xticklabels([])
ax1 = axs[0].twinx()
ax1.yaxis.set_label_position("right")
ax1.set_ylabel('Hourly Resample', rotation = 270, fontsize = 14)
ax1.yaxis.set_label_coords(1.025,0.5)
ax1.set_yticklabels([])


plt.axes(axs[1])
axs[1].plot(Reference_daily.index, Reference_daily.PM2_5, label = 'Unit 1')
axs[1].plot(Paccar_daily.index, Paccar_daily.PM2_5, label = 'Unit 2')
axs[1].plot(Augusta_daily.index, Augusta_daily.PM2_5, label = 'BAM')
axs[1].set_ylim([-5, 50])
axs[1].set_xticklabels([])
axs[1].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
ax2 = axs[1].twinx()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('24 Hr Resample', rotation = 270, fontsize = 14)
ax2.yaxis.set_label_coords(1.025,0.5)
ax2.set_yticklabels([])
axs[1].legend(loc='best', fontsize=12)  
axs[1].set_xticklabels(Reference_daily.index, fontsize = 14)
axs[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator([]))
axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
axs[1].tick_params(axis="y", labelsize=12)
plt.xticks(fontsize = 14)

plt.tight_layout()

#plt.savefig('/Users/matthew/Desktop/thesis/publishing/time_series_gridplot_hourly.png')
plt.savefig('/Users/matthew/Desktop/thesis/publishing/time_series_gridplot_raw_measurements.png')

plt.show()
#%%

plt.rcParams['savefig.facecolor']='white'
#fig, axs = plt.subplots(2, 2, figsize=(8,8), facecolor='white')
fig, axs = plt.subplots(2, 1, figsize=(8,6), facecolor='white')

axs[0].set_title("24 Hr Resample", fontsize = 16)
axs[0].plot(Paccar_daily.index, Paccar_daily.PM2_5)
axs[0].plot(Reference_daily.index, Reference_daily.PM2_5)
axs[0].plot(Augusta_daily.index, Augusta_daily.PM2_5)
axs[0].set_ylim([-5, 50])
axs[0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
#axs[0].yaxis.set_label_position("right")
#axs[0].set_ylabel('Raw Measurements', rotation = 270, fontsize = 14)
#axs[0].yaxis.set_label_coords(1.075,0.5)
axs[0].set_xticklabels([])
axs[0].tick_params(axis="y", labelsize=12)
ax1 = axs[0].twinx()
ax1.yaxis.set_label_position("right")
ax1.set_ylabel('Raw Measurements', rotation = 270, fontsize = 14)
ax1.yaxis.set_label_coords(1.025,0.5)
ax1.set_yticklabels([])


plt.axes(axs[1])
axs[1].plot(Reference_daily.index, Reference_daily.PM2_5_corrected, label = 'Unit 1')
axs[1].plot(Paccar_daily.index, Paccar_daily.PM2_5_corrected, label = 'Unit 2')
axs[1].plot(Augusta_daily.index, Augusta_daily.PM2_5, label = 'BAM')
axs[1].set_ylim([-5, 50])
axs[1].legend(loc='best', fontsize = 12) 
axs[1].set_xticklabels([])
axs[1].set_xticklabels(Reference_daily.index)
axs[1].xaxis.set_major_locator(matplotlib.dates.MonthLocator([]))
axs[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
axs[1].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
#axs[1].yaxis.set_label_position("right")
#axs[1].set_ylabel('Calibrated Measurements', rotation = 270, fontsize = 14)
#axs[1].yaxis.set_label_coords(1.075,0.5)
axs[1].tick_params(axis="y", labelsize=12)
plt.xticks(fontsize = 14)
ax2 = axs[1].twinx()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Raw Measurements', rotation = 270, fontsize = 14)
ax2.yaxis.set_label_coords(1.025,0.5)
ax2.set_yticklabels([])
 
plt.tight_layout()


#plt.savefig('/Users/matthew/Desktop/thesis/publishing/time_series_gridplot.png')
plt.savefig('/Users/matthew/Desktop/thesis/publishing/time_series_gridplot_daily.png')

plt.show()


#%%
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# combine scatters into gridplot

plt.rcParams['savefig.facecolor']='white'
fig, axs = plt.subplots(2, 2, figsize=(8,8), facecolor='white')

plt.axes(axs[0,0])
axs[0, 0].set_title('Hourly Resample', fontsize = 16)
axs[0, 0].scatter(Augusta_hourly.PM2_5, Paccar_hourly.PM2_5)
axs[0, 0].scatter(Augusta_hourly.PM2_5, Reference_hourly.PM2_5)
axs[0, 0].set_xlim([-5, 70])
axs[0, 0].set_ylim([-5, 70])
axs[0, 0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
axs[0, 0].tick_params(axis="y", labelsize=12)
plt.xticks(fontsize = 14)
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = axs[0, 0].transAxes
line.set_transform(transform)
axs[0, 0].add_line(line)

plt.axes(axs[1,0])
axs[1, 0].scatter(Augusta_hourly.PM2_5, Reference_hourly.PM2_5_corrected, label = 'Unit 1')
axs[1, 0].scatter(Augusta_hourly.PM2_5, Paccar_hourly.PM2_5_corrected, label = 'Unit 2')
axs[1, 0].set_ylim([-4, 40]) # -4 is min and 38 is max BAM value (needed so can set scale so line going from corner to corner is a 1 to 1 line)
axs[1, 0].set_xlim([-4, 40])
axs[1, 0].legend(loc='best', fontsize=12)  
axs[1, 0].set_ylabel('PM 2.5 (ug/m³)', fontsize = 14)
axs[1, 0].tick_params(axis="y", labelsize=12)
plt.xticks(fontsize = 14)
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = axs[1, 0].transAxes
line.set_transform(transform)
axs[1, 0].add_line(line)

plt.axes(axs[0,1])
axs[0, 1].set_title("24 Hr Resample", fontsize = 16)
axs[0, 1].scatter(Augusta_daily.PM2_5, Reference_daily.PM2_5, label = 'Unit_1')
axs[0, 1].scatter(Augusta_daily.PM2_5, Paccar_daily.PM2_5, label = 'Unit 2')
axs[0, 1].set_ylim([-1, 50])
axs[0, 1].set_xlim([-1, 50])
axs[0, 1].yaxis.set_label_position("right")
axs[0, 1].set_ylabel('Raw Measurements', rotation = 270, fontsize = 14)
axs[0, 1].yaxis.set_label_coords(1.075,0.5)
axs[0, 1].tick_params(axis="y", labelsize=12)
plt.xticks(fontsize = 14)
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = axs[0, 1].transAxes
line.set_transform(transform)
axs[0, 1].add_line(line)

plt.axes(axs[1,1])
axs[1, 1].scatter(Augusta_daily.PM2_5, Reference_daily.PM2_5_corrected, label = 'Unit 1')
axs[1, 1].scatter(Augusta_daily.PM2_5, Paccar_daily.PM2_5_corrected, label = 'Unit 2')
axs[1, 1].set_ylim([-1, 25])
axs[1, 1].set_xlim([-1, 25])
axs[1, 1].yaxis.set_label_position("right")
axs[1, 1].set_ylabel('Calibrated Measurements', rotation = 270, fontsize = 14)
axs[1, 1].yaxis.set_label_coords(1.075,0.5)
axs[1, 1].tick_params(axis="y", labelsize=12)
axs[1, 1].legend(loc='best', fontsize=12)  
plt.xticks(fontsize = 14)
line = mlines.Line2D([0, 1], [0, 1], color='black')
transform = axs[1, 1].transAxes
line.set_transform(transform)
axs[1, 1].add_line(line)

 
plt.tight_layout()

plt.savefig('/Users/matthew/Desktop/thesis/publishing/scatter_gridplot.png')

plt.show()
