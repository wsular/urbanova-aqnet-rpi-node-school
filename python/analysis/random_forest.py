#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:14:01 2020

@author: matthew
"""

# If have PM10 or not, need to make sure to change line 508 and 245
# lines 208 and 223- 226 control adding wind speed and direction to available features
# line 520 for making sure correct datetime column is selected

import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
import datetime
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import dateutil.parser as parser
from scipy import stats
from scipy import optimize  
import scipy
from sklearn.model_selection import RandomizedSearchCV
from gaussian_fit_function import gaussian_fit
import statsmodels.api as sm
from spec_humid import spec_humid
from load_indoor_data import load_indoor
# get a list of models to evaluate

def get_models1():
	models = dict()
	models['10'] = RandomForestRegressor(max_samples=0.1, random_state = 42)
	models['20'] = RandomForestRegressor(max_samples=0.2, random_state = 42)
	models['30'] = RandomForestRegressor(max_samples=0.3, random_state = 42)
	models['40'] = RandomForestRegressor(max_samples=0.4, random_state = 42)
	models['50'] = RandomForestRegressor(max_samples=0.5, random_state = 42)
	models['60'] = RandomForestRegressor(max_samples=0.6, random_state = 42)
	models['70'] = RandomForestRegressor(max_samples=0.7, random_state = 42)
	models['80'] = RandomForestRegressor(max_samples=0.8, random_state = 42)
	models['90'] = RandomForestRegressor(max_samples=0.9, random_state = 42)
	models['100'] = RandomForestRegressor(max_samples=None, random_state = 42)   # bootstrap sample size is equal to size of training set
	return models
 
# get a list of models to evaluate
def get_models2():
	models = dict()
	models['1'] = RandomForestRegressor(max_features=1, random_state = 42)
	models['2'] = RandomForestRegressor(max_features=2, random_state = 42)
	models['3'] = RandomForestRegressor(max_features=3, random_state = 42)
	models['4'] = RandomForestRegressor(max_features=4, random_state = 42)
	return models


# get a list of models to evaluate
def get_models3():
	models = dict()
	models['10'] = RandomForestRegressor(n_estimators=10, random_state = 42)
	models['50'] = RandomForestRegressor(n_estimators=50, random_state = 42)
	models['100'] = RandomForestRegressor(n_estimators=100, random_state = 42)
	models['500'] = RandomForestRegressor(n_estimators=500, random_state = 42)
	models['1000'] = RandomForestRegressor(n_estimators=1000, random_state = 42)
	return models


# get a list of models to evaluate
def get_models4():
	models = dict()
	models['1'] = RandomForestRegressor(max_depth=1, random_state = 42)
	models['2'] = RandomForestRegressor(max_depth=2, random_state = 42)
	models['3'] = RandomForestRegressor(max_depth=3, random_state = 42)
	models['4'] = RandomForestRegressor(max_depth=4, random_state = 42)
	models['5'] = RandomForestRegressor(max_depth=5, random_state = 42)
	models['6'] = RandomForestRegressor(max_depth=6, random_state = 42)
	models['7'] = RandomForestRegressor(max_depth=7, random_state = 42)
	models['None'] = RandomForestRegressor(max_depth=None, random_state = 42)
	return models

# evaluate a given model using cross-validation
#def evaluate_model(model):
#	cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)   # used k = 5 from Zimmerman paper
#	scores = cross_val_score(model, train_features, train_labels, scoring='r2', cv=cv, n_jobs=-1, error_score='raise') #neg_root_mean_squared_error
#	return scores

def evaluate_model(model):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, train_features, train_labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def evaluate_model1(model):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, train_features, train_labels, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def evaluate_model2(model):
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
    scores = cross_val_score(model, train_features, train_labels, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1, error_score='raise')
    return scores

# get the dataset
#def get_dataset():
#	X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)
#	return X, y
 
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

# Use BAM data from AIRNOW
#files = glob('/Users/matthew/Desktop/data/AirNow/Augusta_AirNow_updated.csv')

# Use BAM data from SRCAA
#files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')

# USE SRCAA data including wind speed and direction
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/All_overlap.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)


inv_df = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/radiosondes/inv_height_m.csv')
files.sort()
for file in files:
    inv_df = pd.concat([inv_df, pd.read_csv(file)], sort=False)

print(inv_df.dtypes)
print(Paccar_All.dtypes)
inv_df['date_obj'] =  pd.to_datetime(inv_df['datetime'])#, format='Y-%m-%dT%H:%M%:%SZ')
print(inv_df.dtypes)

inv_df['iso_date'] = inv_df['date_obj'].apply(lambda x: x.isoformat())
inv_df.index = inv_df.iso_date

#%%
del inv_df['datetime']
del inv_df['date_obj']
del inv_df['iso_date']
#%%
    
# Choose dates of interest
    # Augusta overlap period
start_time = '2019-12-17 15:00' # use 15:00 hrs if not using inv height
end_time = '2020-03-05 23:00'

# dates that Clarity used
# whole range
#start_time = '2019-12-18 00:00'
#end_time = '2020-03-05 23:00'

# Clarity 'Jan'
#start_time = '2019-12-18 00:00'
#end_time = '2020-01-31 23:00'

# Clarity Feb
#start_time = '2020-02-01 00:00'
#end_time = '2020-02-29 23:00'

# March 1-5
#start_time = '2020-03-01 00:00'
#end_time = '2020-03-05 23:00'

interval = '60T'
#interval = '15T'
#interval = '24H'
#%%

#Compare Clarity Units to Augusta SRCAA BAM as time series
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure

inv_df = inv_df.loc[start_time:end_time]
#%%

# Load data used for RF

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time]
Augusta = Augusta.resample(interval).mean()
Augusta['time'] = Augusta.index

Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar1 = Paccar_All.loc[start_time:end_time]
Paccar1 = Paccar1.resample(interval).mean()
Paccar1['BAM'] = Augusta['PM2_5']
###Paccar1['ws'] = Augusta['Wind_speed_mph']   ###

# Merge with inversion dataframe
#Paccar2=pd.merge(Paccar1,inv_df, how='outer', left_index=True, right_index=True)     # use how='inner' to only get matching indices (no NA's)
#Paccar2 = Paccar2[Paccar2['ws'].notna()]

# forward fill inversion column
#Paccar3 = Paccar2.fillna(method='ffill')
#Paccar = Paccar3[Paccar3['inv_height'].notna()]

# fill inversion height using nearest value
#Paccar = Paccar2
#Paccar['inv_height'] = Paccar.interpolate(method='nearest')['inv_height']


Paccar = Paccar1
###Paccar['wind_dir'] = Augusta['Wind_dir_deg']   ###
###Paccar = Paccar[Paccar['ws'].notna()]          ###
###Paccar = Paccar[Paccar['wind_dir'].notna()]    ###
###Paccar = Paccar[Paccar['inv_height'].notna()]

Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference1 = Reference_All.loc[start_time:end_time]
Reference1 = Reference1.resample(interval).mean()
Reference1['BAM'] = Augusta['PM2_5']
Reference1['ws'] = Augusta['Wind_speed_mph']   ###

# Merge with inversion dataframe
#Reference2=pd.merge(Reference1,inv_df, how='outer', left_index=True, right_index=True)     # use how='inner' to only get matching indices (no NA's)
#Reference2 = Reference2[Reference2['ws'].notna()]

# forward fill inversion column
#Reference3 = Reference2.fillna(method='ffill')
#Reference = Reference3[Reference3['inv_height'].notna()]

# fill inversion height using nearest value
#Reference = Reference2
#Reference['inv_height'] = Reference.interpolate(method='nearest')['inv_height']


Reference = Reference1
Reference['wind_dir'] = Augusta['Wind_dir_deg']   ###
Reference = Reference[Reference['ws'].notna()]          ###
Reference = Reference[Reference['wind_dir'].notna()]    ###
###Reference = Reference[Reference['inv_height'].notna()]

#%%
# Load Indoor unit pressure so can calculate specific humidity to determine the effect of using this vs RH on RF model

stevens_bme = pd.DataFrame({})
stevens_bme_json = pd.DataFrame({})
stevens_bme, stevens_bme_json = load_indoor('Stevens', stevens_bme,stevens_bme_json, interval, start_time, end_time)

spec_humid(stevens_bme, stevens_bme_json, Reference)


#%%
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
# k-fold cross validation

###features = Paccar                 ###
###features['datetime'] = Paccar.index   ###

features = Reference
features['datetime'] = Reference.index

print(features.describe())

features = features.dropna() 
labels = np.array(features['BAM'])
features = features.drop('BAM', axis = 1)
features = features.drop('PM10', axis = 1)
#features = features.drop('pressure', axis = 1)    ###  For looking at using specific humidity
#features = features.drop('dewpoint', axis = 1)    ###  For looking at using specific humidity
#features = features.drop('spec_humid', axis = 1)  ###  For looking at using specific humidity
#features = features.drop('Rel_humid', axis = 1)   ###  For looking at using specific humidity
#features = features[['PM2_5', 'spec_humid_unitless', 'temp', 'datetime']]  ###  For looking at using specific humidity
                     ###

feature_list = list(features.columns)
feature_list.pop()          # remove datetime from feature list 
print(features.describe())

features = np.array(features)
#%%
# Split the data into training and testing sets    
train_features1, test_features1, train_labels, test_labels = train_test_split(features, labels, 
                                                                            test_size = 0.30, random_state = 12)  # use 25% of data for test set and 75% for training
#%%
# "1" is for plotting, but need to take it out of the actual RF (just want to be able to match dates if need be)

## #### MAKE SURE THE FOLLOWING DELETED COLUMNS ACTUALLY MATCH UP WITH DATETIME (Might be different based on the number of features used)

train_features = np.delete(train_features1, 5, 1)  # delete datetime column from train_features
test_features = np.delete(test_features1, 5, 1)  # delete datetime column from test_features
#features = Paccar
#features['datetime'] = Paccar.index
# Descriptive statistics for each column
#print(features.describe())

# Labels are the values we want to predict
#labels = np.array(features['BAM'])
# Remove the labels from the features
# axis 1 refers to the columns
#features = features.drop('BAM', axis = 1)
# Saving feature names for later use
#feature_list = list(features.columns)
#feature_list.pop()          # remove datetime from feature list 
# Convert to numpy array
#features = np.array(features)

# Split the data into training and testing sets
#train_features1, test_features1, train_labels, test_labels = train_test_split(features, labels, 
#                                                                            test_size = 0.40, random_state = 2)  # use 25% of data for test set and 75% for training


### This is for a single k-fold cv run. Instead, use the following cells (with the different RF models and kfold cv to find the best parameters and then apply those parameters to this RF model and evaluate this RF model on the training data. (can make runs with different parameters to get best results) then, apply RF to test data and compare performance (make sure it isnt overfitting))
# evaluate the model
#cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
#for train_index, test_index in cv.split(features):
#    print("TRAIN:", train_index, "TEST:", test_index)
    
#n_scores = cross_val_score(model, train_features, train_labels, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
#print('Training MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%%
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
# The example below demonstrates the effect of different bootstrap 
# sample sizes from 10 percent to 100 percent (of the sample size) 
# on the random forest algorithm.

# define dataset
#X, y = get_dataset()
    
# get the models to evaluate
models = get_models1()
# evaluate the models and store results
results, names = list(), list()
results1, names1 = list(), list()
results2, names2 = list(), list()

for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('neg_mean_abs_error          ' + '>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    scores1 = evaluate_model1(model)
    results1.append(scores1)
    print('r^2                         ' + '>%s %.3f (%.3f)' % (name, mean(scores1), std(scores1)))
    scores2 = evaluate_model2(model)
    results2.append(scores2)
    print('neg_root_mean_squared_error ' + '>%s %.3f (%.3f)' % (name, mean(scores2), std(scores2)))
# plot model performance for comparison
fig = plt.figure()
fig.suptitle('Bootstrap Sample Size', fontsize=20)
plt.boxplot(results, labels=names, showmeans=True)
plt.xticks(rotation=45)
plt.xlabel('Samples', fontsize=18)
plt.ylabel('NMAE', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Bootstrap Sample Size', fontsize=20)
plt.boxplot(results1, labels=names, showmeans=True)
plt.xticks(rotation=45)
plt.xlabel('Samples', fontsize=18)
plt.ylabel('r^2', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Bootstrap Sample Size', fontsize=20)
plt.boxplot(results2, labels=names, showmeans=True)
plt.xticks(rotation=45)
plt.xlabel('Samples', fontsize=18)
plt.ylabel('NRMSE', fontsize=16)
plt.show()
#%%
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
# The example below explores the effect of the number of features randomly 
# selected at each split point on model accuracy

# get the models to evaluate
models = get_models2()
# evaluate the models and store results
results, names = list(), list()
results1, names1 = list(), list()
results2, names2 = list(), list()

for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('neg_mean_abs_error          ' + '>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    scores1 = evaluate_model1(model)
    results1.append(scores1)
    print('r^2                         ' + '>%s %.3f (%.3f)' % (name, mean(scores1), std(scores1)))
    scores2 = evaluate_model2(model)
    results2.append(scores2)
    print('neg_root_mean_squared_error ' + '>%s %.3f (%.3f)' % (name, mean(scores2), std(scores2)))
# plot model performance for comparison
fig = plt.figure()
fig.suptitle('Max Features', fontsize=20)
plt.boxplot(results, labels=names, showmeans=True)
plt.xlabel('Features', fontsize=18)
plt.ylabel('NMAE', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Max Features', fontsize=20)
plt.boxplot(results1, labels=names, showmeans=True)
plt.xlabel('Features', fontsize=18)
plt.ylabel('r^2', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Max Features', fontsize=20)
plt.boxplot(results2, labels=names, showmeans=True)
plt.xlabel('Features', fontsize=18)
plt.ylabel('NRMSE', fontsize=16)
plt.show()

#%%
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
# The number of trees can be set via the “n_estimators” argument and defaults to 100.
# The example below explores the effect of the number of trees with values between 10 to 1,000

# get the models to evaluate
models = get_models3()
# evaluate the models and store results
results, names = list(), list()
results1, names1 = list(), list()
results2, names2 = list(), list()

for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('neg_mean_abs_error          ' + '>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    scores1 = evaluate_model1(model)
    results1.append(scores1)
    print('r^2                         ' + '>%s %.3f (%.3f)' % (name, mean(scores1), std(scores1)))
    scores2 = evaluate_model2(model)
    results2.append(scores2)
    print('neg_root_mean_squared_error ' + '>%s %.3f (%.3f)' % (name, mean(scores2), std(scores2)))
# plot model performance for comparison
fig = plt.figure()
fig.suptitle('Number of Trees', fontsize=20)
plt.boxplot(results, labels=names, showmeans=True)
plt.xlabel('Trees', fontsize=18)
plt.ylabel('NMAE', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Number of Trees', fontsize=20)
plt.boxplot(results1, labels=names, showmeans=True)
plt.xlabel('Trees', fontsize=18)
plt.ylabel('r^2', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Number of Trees', fontsize=20)
plt.boxplot(results2, labels=names, showmeans=True)
plt.xlabel('Trees', fontsize=18)
plt.ylabel('NRMSE', fontsize=16)
plt.show()

#%%
# https://machinelearningmastery.com/random-forest-ensemble-in-python/
# The example below explores the effect of random forest maximum 
# tree depth on model performance.

# get the models to evaluate
models = get_models4()
# evaluate the models and store results
results, names = list(), list()
results1, names1 = list(), list()
results2, names2 = list(), list()

for name, model in models.items():
    scores = evaluate_model(model)
    results.append(scores)
    names.append(name)
    print('neg_mean_abs_error          ' + '>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    scores1 = evaluate_model1(model)
    results1.append(scores1)
    print('r^2                         ' + '>%s %.3f (%.3f)' % (name, mean(scores1), std(scores1)))
    scores2 = evaluate_model2(model)
    results2.append(scores2)
    print('neg_root_mean_squared_error ' + '>%s %.3f (%.3f)' % (name, mean(scores2), std(scores2)))
# plot model performance for comparison
fig = plt.figure()
fig.suptitle('Max Depth', fontsize=20)
plt.boxplot(results, labels=names, showmeans=True)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('NMAE', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Max Depth', fontsize=20)
plt.boxplot(results1, labels=names, showmeans=True)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('r^2', fontsize=16)
plt.show()

fig = plt.figure()
fig.suptitle('Max Depth', fontsize=20)
plt.boxplot(results2, labels=names, showmeans=True)
plt.xlabel('Depth', fontsize=18)
plt.ylabel('NRMSE', fontsize=16)
plt.show()

#%%

# Evaluate RF model performance on training data

# define the model
rf  = RandomForestRegressor(n_estimators = 500, random_state = 12, max_samples = 0.7, 
                           max_features = 2, min_samples_leaf = 2, max_depth = 20, min_samples_split=4, bootstrap=True)
rf.fit(train_features, train_labels)

# Use the forest's predict method on the train data
predictions = rf.predict(train_features)
print('Training Data')
print( rf.score(train_features, train_labels))
#print(rf.oob_score(train_features, train_labels))
print('Mean Absolute Error:', metrics.mean_absolute_error(train_labels, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(train_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(train_labels, predictions)))
# Calculate the absolute errors
errors = abs(predictions - train_labels)
# Print out the mean absolute error (mae)
print('Training Mean Absolute Error:', round(np.mean(errors), 2), 'ug/m3')


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
print('Test Data')
print(rf.score(test_features, test_labels))
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Test Mean Absolute Error:', round(np.mean(errors), 2), 'ug/m3')

# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': features[:,3], 'actual': labels})

# Dataframe with predictions and dates
test_dates = test_features1[:,3]
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions, 'BAM': test_labels})
predictions_data = predictions_data.sort_values('date')
predictions_data['prediction_residuals'] = predictions_data['prediction'] - predictions_data['BAM']

predictions_data['Location'] = 'Reference'
#########predictions_data['Location'] = 'Paccar'

predictions_data.index = predictions_data['date']


res_over_5 = abs(predictions_data['prediction_residuals']).values
res_over_5 = res_over_5[res_over_5 >= 5]

count_over_5 = len(res_over_5)

total_count = len(predictions_data['BAM'])

fraction_over = count_over_5/total_count
fraction_under = 1 - fraction_over
print(' Percentage of residuals over 5 ug/m3 = ', fraction_over)
print(' Percentage of residuals under 5ug/m3 = ', fraction_under)


# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#%%
gaussian_fit(predictions_data)

# Check with stats model output

X = predictions_data[['BAM']]#,'Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
#X = X.dropna()
y_ = predictions_data['prediction'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y_, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print_model = model.summary()
print(print_model)

#sigma_i = 2.45*2       # Hourly Reference RF calibrated standard deviation
#sigma_i = 2.47*2       # Hourly Paccar RF calibrated standard deviation

#sigma_i = 1.46*2       # 24 hour Paccar RF calibrated standard deviation
sigma_i = 1.48*2       # 24 hour Reference RF calibrated standard deviation

n = len(predictions_data['prediction'])

# Von's method for error estimation
S = n*(1/(sigma_i**2))
predictions_data['S_x'] = predictions_data['BAM']/(sigma_i**2)
S_x = predictions_data['S_x'].sum()
predictions_data['S_y'] = predictions_data['prediction']/(sigma_i**2)
S_y = predictions_data['S_y'].sum()
predictions_data['S_xx'] = (predictions_data['BAM']**2)/(sigma_i**2)
S_xx = predictions_data['S_xx'].sum()
predictions_data['S_xy'] = ((predictions_data['prediction']*predictions_data['BAM'])/sigma_i**2)
S_xy = predictions_data['S_xy'].sum()
delta = S*S_xx - (S_x)**2
a = ((S_xx*S_y) - (S_x*S_xy))/delta
b = ((S*S_xy) - (S_x*S_y))/delta
var_a = S_xx/delta
var_b = S/delta
stdev_a = var_a**0.5
stdev_b = var_b**0.5
se_a = stdev_a/(n**0.5)
se_b = stdev_b/(n**0.5)
r_ab = (-1*S_x)/((S*S_xx)**0.5)

print('predictions_data a =', a, '\n',
      'predictions_data b =', b, '\n')

print('predictions_data var a =', var_a, '\n',
      'predictions_data var b =', var_b, '\n')

print('predictions_data standard dev a =', stdev_a, '\n',
      'predictions_data standard dev b =', stdev_b, '\n')

print('predictions_data standard error a =', se_a, '\n',
      'predictions_data standard error b =', se_b, '\n',
      'predictions_data r value =', r_ab)
#%%
# Calculate mean absolute percentage error (MAPE)
#mape = 100 * (errors / test_labels)
# Calculate and display accuracy
#print(mape)
#print(np.mean(mape))
#accuracy = 100 - np.mean(mape)
#print('Accuracy:', round(accuracy, 2), '%')

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


#%%

# Plotting

# Set the style
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# Plot the actual values
plt.plot(true_data['date'], true_data['actual'], 'b-', label = 'actual')
# Plot the predicted values
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label = 'prediction')
plt.xticks(rotation = '60'); 
plt.legend()
# Graph labels
plt.xlabel('Date'); plt.ylabel('PM 2.5 (ug/m^3)'); plt.title('Actual and Predicted Values');

#%%
# Calc best fit line for RF predictions for Paccar node vs Augusta

#the data
x1=np.array(predictions_data.BAM)
y1=np.array(predictions_data.prediction) 
slope11, intercept11, r_value11, p_value11, std_err11 = scipy.stats.linregress(x1, y1)
r_squared1 = r_value11**2

# determine best fit line
par = np.polyfit(x1, y1, 1, full=True)
slope1=par[0][0]
intercept1=par[0][1]
y1_predicted = [slope1*i + intercept1  for i in x1]
#%%
# Plotting

p1 = figure(plot_width=900,
            plot_height=450,
            x_axis_type='datetime',
            x_axis_label='Date (local)',
            y_axis_label='PM 2.5 (ug/m3)')
p1.title.text = 'RF Calibration' 
#p1.line(true_data.date,         true_data.actual,             legend='BAM Measurements',    color='gold',     line_width=2,       muted_color='gold', muted_alpha=0.1)
p1.line(predictions_data.date,  predictions_data.prediction,  legend='RF Calibrated',         color='black',    line_alpha = 0.9,   line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(predictions_data.date,  predictions_data.BAM,         legend='BAM ',        color='red',      line_alpha = 0.3,   line_width=2, muted_color='red', muted_alpha=0.2)

p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Augusta BAM and RF Predictions")

p2 = figure(plot_width=900,
            plot_height=450,
           # x_axis_type='datetime',
            x_axis_label='BAM (ug/m3)',
            y_axis_label='RF Predictions (ug/m3)')
p2.title.text = 'RF Add. Predictions' 
p2.scatter(predictions_data.BAM,            predictions_data.prediction,             legend='BAM vs RF Predictions',      color='black',     line_width=2)
#p2.circle(df.Augusta, df.Paccar, legend='Paccar', color='blue')
p2.line(x1,y1_predicted,color='black',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))
p2.legend.location='top_left'

tab2 = Panel(child=p2, title="Augusta BAM vs RF Predictions")


tabs = Tabs(tabs=[ tab1, tab2])

show(tabs)


#%%

# The following script divides data into attributes and labels

X = Paccar.iloc[:, 0:4].values
y = Paccar.iloc[:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

# Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#%%

# Training the algorithm

regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# evaluating the algorithm
#%%
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#%%

from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=2)
# define the model
model = RandomForestRegressor()
# evaluate the model
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%%
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
# Single random forest

#features = Paccar
#features['datetime'] = Paccar.index
# Descriptive statistics for each column
#print(features.describe())

# Labels are the values we want to predict
#labels = np.array(features['BAM'])
# Remove the labels from the features
# axis 1 refers to the columns
#features = features.drop('BAM', axis = 1)
# Saving feature names for later use
#feature_list = list(features.columns)
#feature_list.pop()          # remove datetime from feature list 
# Convert to numpy array
#features = np.array(features)

# Split the data into training and testing sets
#train_features1, test_features1, train_labels, test_labels = train_test_split(features, labels, 
#                                                                            test_size = 0.40, random_state = 2)  # use 25% of data for test set and 75% for training

#print('Training Features Shape:', train_features1.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features1.shape)
#print('Testing Labels Shape:', test_labels.shape)

#train_features = np.delete(train_features1, 4, 1)  # delete datetime column from train_features
#test_features = np.delete(test_features1, 4, 1)  # delete datetime column from test_features

#print('Training Features Shape:', train_features.shape)
#print('Training Labels Shape:', train_labels.shape)
#print('Testing Features Shape:', test_features.shape)
#print('Testing Labels Shape:', test_labels.shape)

#%%
# Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 2, max_features = 'sqrt')
# Train the model on training data
#rf.fit(train_features, train_labels)


# Use the forest's predict method on the test data
#predictions = rf.predict(test_features)
# Calculate the absolute errors
#errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'ug/m3')
