#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:13:12 2020

@author: matthew
"""
import pandas as pd
from glob import glob

adams = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Adams/WSU*.json')
files.sort()
for file in files:
    adams = pd.concat([adams, pd.read_json(file)], sort=False)
adams['Datetime'] = pd.to_datetime(adams['Datetime'])
adams = adams.sort_values('Datetime')
adams.index = adams.Datetime


audubon = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Audubon/WSU*.json')
files.sort()
for file in files:
    audubon = pd.concat([audubon, pd.read_json(file)], sort=False)
audubon['Datetime'] = pd.to_datetime(audubon['Datetime'])
audubon = audubon.sort_values('Datetime')
audubon.index = audubon.Datetime


balboa = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Balboa/WSU*.json')
files.sort()
for file in files:
    balboa = pd.concat([balboa, pd.read_json(file)], sort=False)
balboa['Datetime'] = pd.to_datetime(balboa['Datetime'])
balboa = balboa.sort_values('Datetime')
balboa.index = balboa.Datetime


browne = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Browne/WSU*.json')
files.sort()
for file in files:
    browne = pd.concat([browne, pd.read_json(file)], sort=False)
browne['Datetime'] = pd.to_datetime(browne['Datetime'])
browne = browne.sort_values('Datetime')
browne.index = browne.Datetime


grant = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Grant/WSU*.json')
files.sort()
for file in files:
    grant = pd.concat([grant, pd.read_json(file)], sort=False)
grant['Datetime'] = pd.to_datetime(grant['Datetime'])
grant = grant.sort_values('Datetime')
grant.index = grant.Datetime


jefferson = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Jefferson/WSU*.json')
files.sort()
for file in files:
    jefferson = pd.concat([jefferson, pd.read_json(file)], sort=False)
jefferson['Datetime'] = pd.to_datetime(jefferson['Datetime'])
jefferson = jefferson.sort_values('Datetime')
jefferson.index = jefferson.Datetime


lidgerwood = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Lidgerwood/WSU*.json')
files.sort()
for file in files:
    lidgerwood = pd.concat([lidgerwood, pd.read_json(file)], sort=False)
lidgerwood['Datetime'] = pd.to_datetime(lidgerwood['Datetime'])
lidgerwood = lidgerwood.sort_values('Datetime')
lidgerwood.index = lidgerwood.Datetime


regal = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Regal/WSU*.json')
files.sort()
for file in files:
    regal = pd.concat([regal, pd.read_json(file)], sort=False)
regal['Datetime'] = pd.to_datetime(regal['Datetime'])
regal = regal.sort_values('Datetime')
regal.index = regal.Datetime


sheridan = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Sheridan/WSU*.json')
files.sort()
for file in files:
    sheridan = pd.concat([sheridan, pd.read_json(file)], sort=False)
sheridan['Datetime'] = pd.to_datetime(sheridan['Datetime'])
sheridan = sheridan.sort_values('Datetime')
sheridan.index = sheridan.Datetime


stevens = pd.DataFrame({})
files   = glob('/Users/matthew/Desktop/data/urbanova/ramboll/Stevens/WSU*.json')
files.sort()
for file in files:
    stevens = pd.concat([stevens, pd.read_json(file)], sort=False)
stevens['Datetime'] = pd.to_datetime(stevens['Datetime'])
stevens = stevens.sort_values('Datetime')
stevens.index = stevens.Datetime
