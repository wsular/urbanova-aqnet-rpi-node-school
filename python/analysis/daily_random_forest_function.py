#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:57:10 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:01:43 2020

@author: matthew
"""

import pandas as pd
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import scipy
from bokeh.models import Panel, Tabs
from bokeh.plotting import figure
from gaussian_fit_function import gaussian_fit
from bokeh.io import output_notebook, output_file, show
import statsmodels.api as sm
#%%


def daily_random_forest(model, location):
    features = location
    #features['datetime'] = location.index
    print(features.describe())
  #  features = features.drop('lower_uncertainty', axis = 1)
   # features = features.drop('upper_uncertainty', axis = 1)
    features = features.drop('PM2_5', axis = 1)   # want to use the PM2_5_corrected that has been adjusted to the Ref Clarity unit already
    features = features.drop('PM10', axis = 1)
    features = features.drop('ID', axis = 1)
    features = features.drop('Location', axis = 1)
    features = features.drop('time', axis = 1)
    
    print(features.head(10))
    features.rename(columns={'PM2_5_corrected':'PM2_5'}, inplace=True)   # rename so same headers as rf trained on
    features = features[['PM2_5', 'Rel_humid', 'temp']]  # reorder column so same order as rf trained on
    print(features.describe())
    #features = features.dropna()
    print(features.head(10))
    features = np.array(features)
    
    predictions = model.predict(features)
    location['PM2_5_corrected'] = predictions
    
    return location


#%%
    
start_time = '2019-12-17 15:00' # use 15:00 hrs if not using inv height
end_time = '2020-03-05 23:00'

#interval = '60T'
interval = '24H'
#%%

Augusta_All = pd.DataFrame({})

files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/All_overlap.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)
    

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time]
Augusta = Augusta.resample(interval).mean()


Paccar_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Paccar*.csv')
files.sort()
for file in files:
    Paccar_All = pd.concat([Paccar_All, pd.read_csv(file)], sort=False)
    

Paccar_All['time'] = pd.to_datetime(Paccar_All['time'])
Paccar_All = Paccar_All.sort_values('time')
Paccar_All.index = Paccar_All.time
Paccar1 = Paccar_All.loc[start_time:end_time]
Paccar1 = Paccar1.resample(interval).mean()
Paccar1['BAM'] = Augusta['PM2_5']

Paccar = Paccar1


Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv')
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)
    

Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference1 = Reference_All.loc[start_time:end_time]
Reference1 = Reference1.resample(interval).mean()
Reference1['BAM'] = Augusta['PM2_5']

Reference = Reference1

###############features = Reference
features = Paccar#######################
##############features['datetime'] = Reference.index
features['datetime'] = Paccar.index###############

print(features.describe())
labels = np.array(features['BAM'])
features = features.drop('BAM', axis = 1)
features = features.drop('PM10', axis = 1)
features = features.drop('datetime', axis = 1)
feature_list = list(features.columns)
print(features.describe())

features = np.array(features)
#%%

# "1" is for plotting, but need to take it out of the actual RF (just want to be able to match dates if need be)

## #### MAKE SURE THE FOLLOWING DELETED COLUMNS ACTUALLY MATCH UP WITH DATETIME (Might be different based on the number of features used)

# define the model
daily_rf  = RandomForestRegressor(n_estimators = 500, random_state = 12, max_samples = 0.7, 
                           max_features = 2, min_samples_leaf = 2, max_depth = 20, min_samples_split=4, bootstrap=True)
daily_rf.fit(features, labels)

# Use the forest's predict method on the train data
predictions = daily_rf.predict(features)
print('Training Data')
print( daily_rf.score(features, labels))
#print(rf.oob_score(train_features, train_labels))
print('Mean Absolute Error:', metrics.mean_absolute_error(labels, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(labels, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels, predictions)))
# Calculate the absolute errors
errors = abs(predictions - labels)
# Print out the mean absolute error (mae)
print('Training Mean Absolute Error:', round(np.mean(errors), 2), 'ug/m3')


# Dataframe with true values and dates
true_data = pd.DataFrame(data = {'date': Reference.index, 'actual': labels})

# Dataframe with predictions and dates

test_dates = Paccar['datetime']##############

############test_dates = Reference['datetime']############


predictions_data = pd.DataFrame(data = {'date': test_dates, 'prediction': predictions, 'BAM': labels})
predictions_data = predictions_data.sort_values('date')
predictions_data['prediction_residuals'] = predictions_data['prediction'] - predictions_data['BAM']

#############predictions_data['Location'] = 'Reference'###############

predictions_data['Location'] = 'Paccar'#################

predictions_data.index = predictions_data['date']


res_over_5 = abs(predictions_data['prediction_residuals']).values
res_over_5 = res_over_5[res_over_5 >= 5]

count_over_5 = len(res_over_5)

total_count = len(predictions_data['BAM'])

fraction_over = count_over_5/total_count
fraction_under = 1 - fraction_over
print(' Percentage of residuals over 5 ug/m3 = ', fraction_over)
print(' Percentage of residuals under 5 ug/m3 = ', fraction_under)


# Get numerical feature importances
importances = list(daily_rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
#%%
gaussian_fit(predictions_data)
#%%
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

#sigma_i = 1.08*2       # Reference RF calibrated standard deviation for 24 hr avg and entire overlap dataset
sigma_i = 1.05       # Paccar RF calibrated standard deviation for 24 hr avg and entire overlap dataset

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