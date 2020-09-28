#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:32:39 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:14:01 2020

@author: matthew
"""

# If have PM10 or not, need to make sure to change line 508 and 245

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
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, output_file, show
from bokeh.plotting import figure
from pprint import pprint
from sklearn.model_selection import GridSearchCV



def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features) 
   # errors = abs(predictions - test_labels)               # Trying to calculate MAPE results in a divide by 0 so cant use it
    #mape = 100 * np.mean(errors / test_labels)
    #accuracy = 100 - mape
    print('Model Performance on Test Data')
   # print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
   # print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('RF r^2 value:', model.score(test_features, test_labels))
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, predictions))
    mae = metrics.mean_absolute_error(test_labels, predictions)
    print('Mean Squared Error:', metrics.mean_squared_error(test_labels, predictions))
    print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(test_labels, predictions)))
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Test Mean Absolute Error:', round(np.mean(errors), 2), 'ug/m3')
    
    return mae
    #return accuracy
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



#%%

# Resample data to 1 hr intervals

inv_df = inv_df.loc[start_time:end_time]

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time]


Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar1 = Paccar_All.loc[start_time:end_time]
Paccar1 = Paccar1.resample(interval).mean()
Paccar1['BAM'] = Augusta['PM2_5']
###Paccar1['ws'] = Augusta['Wind_speed_mph']

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
###Paccar['wind_dir'] = Augusta['Wind_dir_deg']
###Paccar = Paccar[Paccar['ws'].notna()]
###Paccar = Paccar[Paccar['wind_dir'].notna()]
###Paccar = Paccar[Paccar['inv_height'].notna()]

Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]
Reference = Reference.resample(interval).mean()

#%%

# Split data into features and labels

features = Paccar
features['datetime'] = Paccar.index

print(features.describe())
labels = np.array(features['BAM'])
features = features.drop('BAM', axis = 1)
features = features.drop('PM10', axis = 1)
feature_list = list(features.columns)
feature_list.pop()          # remove datetime from feature list 
print(features.describe())

features = np.array(features)

# Split the data into training and testing sets    
train_features1, test_features1, train_labels, test_labels = train_test_split(features, labels, 
                                                                            test_size = 0.30, random_state = 12)  # use 25% of data for test set and 75% for training
#%%
# "1" is for plotting (still has datetime column so can put in time series), but need to take it out of the actual RF (just want to be able to match dates if need be)

## #### MAKE SURE THE FOLLOWING DELETED COLUMNS ACTUALLY MATCH UP WITH DATETIME (Might be different based on the number of features used)

train_features = np.delete(train_features1, 3, 1)  # delete datetime column from train_features
test_features = np.delete(test_features1, 3, 1)  # delete datetime column from test_features


#%%
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# Large Random Hyperparameter Grid

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 2000, num = 100)]
# Number of features to consider at every split
max_features = [1, 2, 3]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(4, 110, num = 20)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 20, num = 10)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Method of selecting samples for training each tree
bootstrap = [True]
# Percentage of test samples used in each tree bootstrap selection
max_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'max_samples': max_samples}
pprint(random_grid)

#%%

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 10000 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_absolute_error', n_iter = 10000, 
                               cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(train_features, train_labels)
#%%
# Create dataframe with metrics of every single combination tested
random_results = pd.DataFrame(rf_random.cv_results_)
random_results = random_results.sort_values('rank_test_score')
random_results.to_json(r'/Users/matthew/Desktop/data/RF_randomized_search_CV_mae.json')
    
# set RF with best parameters from random sampling
print(rf_random.best_params_)

best_random = rf_random.best_estimator_
# Fit best random model parameters to training data
best_random.fit(train_features, train_labels)

# Evaluate the best random model parameters on the test data
random_accuracy = evaluate(best_random, test_features, test_labels)


#%%

# Grid search with Cross validation (testing every combination of model parameters, not random assortment as above)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 5)]
# Number of features to consider at every split
max_features = [2]
# Maximum number of levels in tree
max_depth = [5,10,15,20]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 20, num = 10)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4]
# Method of selecting samples for training each tree
bootstrap = [True]
# Percentage of test samples used in each tree bootstrap selection
max_samples = [ 0.2, 0.3, 0.4, 0.5]
# Create the random grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'max_samples': max_samples}
pprint(param_grid)
#%%
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, scoring = 'neg_mean_absolute_error',  
                               cv = 3, verbose=2, n_jobs = -1)

# Fit the grid search to the data
grid_search.fit(train_features, train_labels)

#%%
# Create dataframe with metrics of every single combination tested
grid_results = pd.DataFrame(grid_search.cv_results_)
grid_results = grid_results.sort_values('rank_test_score')
grid_results.to_json(r'/Users/matthew/Desktop/data/RF_grid_search_CV_mae.json')
    
# set RF with best parameters from random sampling
print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
# Fit best random model parameters to training data
best_grid.fit(train_features, train_labels)

# Evaluate the best random model parameters on the test data
grid_accuracy = evaluate(best_grid, test_features, test_labels)

#%%
# Best parameters based on grid search cv

best_model = RandomForestRegressor(n_estimators = 500, random_state = 12, max_samples = 0.7, 
                           max_features = 2, min_samples_leaf = 2, max_depth = 20, min_samples_split=4, bootstrap=True)
best_model.fit(train_features, train_labels)
best_accuracy = evaluate(best_model, test_features, test_labels)
#%%

# Compare default rf with optimized random parameters
# If mae is the evaluation parameter, then positive "improvement" percentage means the base model was better

base_model = RandomForestRegressor(n_estimators = 1000, random_state = 12, max_samples = 0.4, 
                           max_features = 2, min_samples_leaf = 1, oob_score = True) #, max_depth = 5

base_model.fit(train_features, train_labels)

base_accuracy = evaluate(base_model, test_features, test_labels)

print('Random Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))
print('Grid Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


#%%

# Augusta data for test dataset
true_data = pd.DataFrame(data = {'date': Paccar.index, 'actual': labels})

# Dataframe with predictions and dates

#predictions = base_model.predict(test_features)
#predictions = best_grid.predict(test_features)
predictions = best_model.predict(test_features)

test_dates = test_features1[:,3]
predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions, 'BAM': test_labels})
predictions_data = predictions_data.sort_values('date')
predictions_data.index = predictions_data['date']

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

p1.line(true_data.date,         true_data.actual,             legend='BAM Measurements',    color='gold',     line_width=2,       muted_color='gold', muted_alpha=0.1)
p1.line(predictions_data.date,  predictions_data.prediction,  legend='Predictions',         color='black',    line_alpha = 0.9,   line_width=2, muted_color='black', muted_alpha=0.2)
p1.line(predictions_data.date,  predictions_data.BAM,         legend='BAM Matching',        color='red',      line_alpha = 0.9,   line_width=2, muted_color='red', muted_alpha=0.2)

p1.legend.click_policy="mute"

tab1 = Panel(child=p1, title="Augusta BAM and RF Predictions")

p2 = figure(plot_width=900,
            plot_height=450,
           # x_axis_type='datetime',
            x_axis_label='BAM (ug/m3)',
            y_axis_label='RF Predictions (ug/m3)')

p2.scatter(predictions_data.BAM,            predictions_data.prediction,             legend='BAM vs RF Predictions',      color='black',     line_width=2)
#p2.circle(df.Augusta, df.Paccar, legend='Paccar', color='blue')
p2.line(x1,y1_predicted,color='black',legend='y='+str(round(slope1,2))+'x+'+str(round(intercept1,2))+ '  ' + 'r^2 = ' + str(round(r_squared1,3)))
p2.legend.location='top_left'

tab2 = Panel(child=p2, title="Augusta BAM vs RF Predictions")

tabs = Tabs(tabs=[ tab1, tab2])

show(tabs)
#%%

p3 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Combination Rank',
            y_axis_label='Negative MAE (ug/m3')

p3.line(random_results.rank_test_score,         random_results.mean_test_score,             legend='RandomSearchCV results',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)

tab3 = Panel(child=p3, title="RF performance")

p4 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='RF max depth',
            y_axis_label='Negative MAE (ug/m3')

p4.title.text = 'Max Depth'    
#p4.scatter(grid_results.param_max_depth,         grid_results.mean_test_score,               color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(grid_results.param_n_estimators,         grid_results.mean_test_score,             legend='n_estimators',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(grid_results.param_min_samples_split,         grid_results.mean_test_score,             legend='min samples split',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(grid_results.param_min_samples_leaf,         grid_results.mean_test_score,             legend='min samples leaf',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(grid_results.param_max_samples,         grid_results.mean_test_score,             legend='max samples',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)

p4.scatter(random_results.param_max_depth,         random_results.mean_test_score,                color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(random_results.param_n_estimators,         random_results.mean_test_score,             legend='n_estimators',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(random_results.param_min_samples_split,         random_results.mean_test_score,             legend='min samples split',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(random_results.param_min_samples_leaf,         random_results.mean_test_score,             legend='min samples leaf',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
#p4.scatter(random_results.param_max_samples,         random_results.mean_test_score,             legend='max samples',    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)

tab4 = Panel(child=p4, title="max depth")

p5 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Number of Trees ',
            y_axis_label='Negative MAE (ug/m3')
p5.title.text = 'Number of Trees'    
#p5.scatter(grid_results.param_n_estimators,         grid_results.mean_test_score,         color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
p5.scatter(random_results.param_n_estimators,         random_results.mean_test_score,    color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)


tab5 = Panel(child=p5, title="n_estimators")


p6 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Min Sample Split',
            y_axis_label='Negative MAE (ug/m3')
p6.title.text = 'Min Samples for Split'    
#p6.scatter(grid_results.param_min_samples_split,         grid_results.mean_test_score,       color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
p6.scatter(random_results.param_min_samples_split,         random_results.mean_test_score,        color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)


tab6 = Panel(child=p6, title="Min Sample Split")

p7 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Min Samples Leaf',
            y_axis_label='Negative MAE (ug/m3')
p7.title.text = 'Min Samples for Leaf'    
#p7.scatter(grid_results.param_min_samples_leaf,         grid_results.mean_test_score,              color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
p7.scatter(random_results.param_min_samples_leaf,         random_results.mean_test_score,              color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)


tab7 = Panel(child=p7, title="Min Samples Leaf")

p8 = figure(plot_width=900,
            plot_height=450,
          #  x_axis_type='datetime',
            x_axis_label='Max Samples',
            y_axis_label='Negative MAE (ug/m3')
p1.title.text = 'Bootstrap Sample'    
#p8.scatter(grid_results.param_max_samples,         grid_results.mean_test_score,           color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)
p8.scatter(random_results.param_max_samples,         random_results.mean_test_score,            color='black',     line_width=2,       muted_color='gold', muted_alpha=0.1)


tab8 = Panel(child=p8, title="Max Samples")

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8])

show(tabs)







