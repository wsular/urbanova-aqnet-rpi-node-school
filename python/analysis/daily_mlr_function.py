#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:09:21 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:42:25 2020

@author: matthew
"""

import pandas as pd
from glob import glob
import statsmodels.api as sm

#%%

def daily_mlr_function(daily_mlr_model,location):

    X = location[['PM2_5_corrected','Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
    X = X.dropna()
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
  #  mlr_model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    # Note the difference in argument order
    predictions = daily_mlr_model.predict(X)

    # Print out the statistics
    print_model = daily_mlr_model.summary()
    print(print_model)
    location['PM2_5_corrected'] = predictions
    
    return location


#%%
    
Reference_All = pd.DataFrame({})
files = glob('/Users/matthew/Desktop/data/Clarity_Backup/Reference*.csv')
files.sort()
for file in files:
    Reference_All = pd.concat([Reference_All, pd.read_csv(file)], sort=False)
    
#Read in SRCAA Augusta site BAM data
        
Augusta_All = pd.DataFrame({})
        
# Use BAM data from AIRNOW
#files = glob('/Users/matthew/Desktop/data/AirNow/Augusta_AirNow_updated.csv')
        
# Use BAM data from SRCAA
files = glob('/Users/matthew/Desktop/data/SRCAA_Augusta_BAM/Spokane_Augusta*.csv')
files.sort()
for file in files:
    Augusta_All = pd.concat([Augusta_All, pd.read_csv(file)], sort=False)

# Choose dates of interest
    # Augusta overlap period
start_time = '2019-12-17 15:00'
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
    
#interval = '60T'
#interval = '15T'
interval = '24H'

Augusta_All['time'] = pd.to_datetime(Augusta_All['time'])
Augusta_All = Augusta_All.sort_values('time')
Augusta_All.index = Augusta_All.time
Augusta = Augusta_All.loc[start_time:end_time]
Augusta = Augusta.resample(interval).mean()
#Augusta = Augusta.interpolate(method='linear')

    
Reference_All['time'] = pd.to_datetime(Reference_All['time'])
Reference_All = Reference_All.sort_values('time')
Reference_All.index = Reference_All.time
Reference = Reference_All.loc[start_time:end_time]
Reference = Reference.resample(interval).mean()
    
Reference = Reference.dropna()
    
    
Reference['Augusta_PM2_5'] = Augusta['PM2_5']
    
Reference = Reference.dropna()
X = Reference[['PM2_5','Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
    
y = Reference['Augusta_PM2_5']
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
    
# Note the difference in argument order
daily_mlr_model = sm.OLS(y, X).fit() ## sm.OLS(output, inp

