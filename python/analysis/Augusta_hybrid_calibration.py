#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:44:25 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:55:44 2020

@author: matthew
"""

import pandas as pd
from glob import glob
import statsmodels.api as sm
import numpy as np

#%%

def hybrid_function(rf_model, mlr_model,location, name):

    
    df1 = location[location.PM2_5 >= 25]
    df2 = location[location.PM2_5 < 25]

   # print(df1.head(10))
   # print(df2.head(10)) 
    
    X = df1[['PM2_5','Rel_humid', 'temp']] ## X usually means our input variables (or independent variables)
    X = X.dropna()
    X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
  #  mlr_model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
    # Note the difference in argument order
    predictions1 = mlr_model.predict(X)
    df1['PM2_5_corrected'] = predictions1
    # Print out the statistics
    print_model = mlr_model.summary()
    print('1')
    
    
    features = df2
    #features['datetime'] = location.index
    print(features.describe())
    features = features.drop('PM10', axis = 1)   # want to use the PM2_5_corrected that has been adjusted to the Ref Clarity unit already

    
    print(features.head(10))
    features = features[['PM2_5', 'Rel_humid', 'temp']]  # reorder column so same order as rf trained on
    print(features.describe())
    #features = features.dropna()
    print(features.head(10))
    features = np.array(features)
    
    predictions2 = rf_model.predict(features)
    df2['PM2_5_corrected'] = predictions2
    
    combined_df = df1.append(df2)
    location = combined_df
    

    
    return location