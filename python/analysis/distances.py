#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:18:15 2019

@author: matthew
"""

# Find distances between all Clarity sites
import pandas as pd
from geopy.distance import distance



school_names = ['Adams', 
                'Jefferson', 
                'Grant',
                'Sheridan', 
                'Reference',
                'Stevens',
                'Regal', 
                'Audubon', 
                'Lidgerwood', 
                'Browne',
                'Balboa']

Distances = pd.DataFrame()

for name in school_names:
    Distances[name] = name


other_schools= {
    'Adams':(47.621172,  -117.367725),
    'Jefferson': (47.621533,  -117.4098417),
    'Grant': (47.6467083,  -117.390983),
    'Sheridan': (47.6522472, -117.355561),
    'Reference': (47.6608,  -117.4045056),
    'Stevens': (47.671256,  -117.3846583),
    'Regal': (47.69735,  -117.369972),
    'Audubon': (47.6798472,  -117.441739),
    'Lidgerwood': (47.7081417,  -117.405161),
    'Browne': (47.70415, -117.4640639),
    'Balboa': (47.71818056,  -117.4560056),
   }


for school in school_names:
    d = []
    primary_location = school
    print(primary_location)
    
    for school, coord in other_schools.items():
    
        #d.append(distance(Adams_coord, coord))
        dist = distance(other_schools[primary_location], coord).km  #change to .m to put in meters
        d.append(dist)             
        print(school, dist)
    Distances[primary_location] = d

Distances.index = school_names

