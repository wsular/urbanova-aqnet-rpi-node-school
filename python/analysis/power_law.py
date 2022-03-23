#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:49:21 2021

@author: matthew
"""

import numpy as np

def power_law(x, a, b):
    return a*np.power(x, b)
def power_law_cal(y, a , b):
    return (y/a)**(1/b)