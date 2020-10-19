#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:44:34 2020

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 16:59:33 2020

@author: matthew
"""

import statsmodels.api as sm


def mlr_function_general(model, location, location_name):
    
    
    if location_name == 'Audubon':
    
        X = location[['Audubon','Audubon_rh', 'Audubon_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)


    if location_name == 'Adams':
    
        X = location[['Adams','Adams_rh', 'Adams_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)
        
    if location_name == 'Balboa':
    
        X = location[['Balboa','Balboa_rh', 'Balboa_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)
        

    if location_name == 'Browne':
    
        X = location[['Browne','Browne_rh', 'Browne_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)


    if location_name == 'Grant':
    
        X = location[['Grant','Grant_rh', 'Grant_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)

    
    if location_name == 'Jefferson':
    
        X = location[['Jefferson','Jefferson_rh', 'Jefferson_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)


    if location_name == 'Lidgerwood':
    
        X = location[['Lidgerwood','Lidgerwood_rh', 'Lidgerwood_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)


    if location_name == 'Regal':
    
        X = location[['Regal','Regal_rh', 'Regal_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)

    
    if location_name == 'Sheridan':
    
        X = location[['Sheridan','Sheridan_rh', 'Sheridan_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)


    if location_name == 'Stevens':
    
        X = location[['Stevens','Stevens_rh', 'Stevens_temp']] ## X usually means our input variables (or independent variables)  Rel_humid
        X = X.dropna()
        X = sm.add_constant(X)
        predictions = model.predict(X)


    return predictions
