#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 13:40:26 2020

@author: matthew
"""

import smtplib
import json
import datetime

def mail_alert1():                           
    fromaddr = email
    toaddrs  = email
    msg = 'Subject: {}\n\n{}'.format('Indoor Unit ' + sensorParameters['ID'] + ' ' +'Reboot', 'Sensor' + '_' + sensorParameters['ID'] + ' Reboot at ' + currentTime.strftime('%Y%m%d_%H%M%S'))

# Credentials (if needed)
    username = email
    password = pw

# The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()

def mail_alert2():                           
    fromaddr = email
    toaddrs  = email_2
    msg = 'Subject: {}\n\n{}'.format('Indoor Unit ' + sensorParameters['ID'] + ' ' +'Reboot', 'Sensor' + '_' + sensorParameters['ID'] + ' Reboot at ' + currentTime.strftime('%Y%m%d_%H%M%S'))

# Credentials (if needed)
    username = email
    password = pw

# The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()



with open('sensorParameters.json') as json_file:
    sensorParameters=json.load(json_file)

with open('/home/pi/SpokaneSchools/software/Name_1.txt','r') as file:
    email=file.read()
    
with open('/home/pi/SpokaneSchools/software/Name_3.txt','r') as file:
    email_2=file.read()

with open('/home/pi/SpokaneSchools/software/Name_2.txt','r') as file:
    pw=file.read()
    

currentTime = datetime.datetime.now()

mail_alert1()
mail_alert2()