#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:09:46 2020

@author: matthew
"""
from glob import glob
import imageio

#%%


files   = glob('/Users/matthew/Desktop/IDW/png/10_9_to_10_17_19/png*.png')
files.sort()
#%%
images = []
for filename in files:
    images.append(imageio.imread(filename))
imageio.mimsave('/Users/matthew/Desktop/IDW/gif/test_0_1_sec_10_9_to_10_17_19.gif', images, duration=0.1)