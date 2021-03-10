#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 14:49:37 2021

@author: matthew
"""



def remove_smoke(outdoor):
    
    outdoor_start_1 = '2019-09-01 00:00'
    outdoor_end_1 = '2020-09-10 23:00'
    
    outdoor_start_2 = '2020-09-21 20:00'
    outdoor_end_2 = '2021-02-21 00:00'
    
    outdoor_1 = outdoor.loc[outdoor_start_1:outdoor_end_1]
    outdoor_2 = outdoor.loc[outdoor_start_2:outdoor_end_2]
    
    outdoor_cut = outdoor_1.append(outdoor_2)
    outdoor_cut = outdoor_cut.sort_index()
    
    return outdoor_cut