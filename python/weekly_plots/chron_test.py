#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:02:55 2022

@author: matthew
"""

from datetime import datetime
import os
sysDate = datetime.now()
# convert system date to string
curSysDate = (str(sysDate))
# parse the date only from the string
curDate = curSysDate[0:10]
# parse the hour only from the string
curHour = curSysDate[11:13]
# parse the minutes only from the string
curMin = curSysDate[14:16]
# concatenate hour and minute with underscore
curTime = curHour + '_' + curMin
# val for the folder name
folderName =  curDate + '_' + curTime

#parent dir
parent_dir = '/Users/matthew/Desktop/chron_test/'

path = os.path.join(parent_dir, folderName)
# make a directory

os.mkdir(path)