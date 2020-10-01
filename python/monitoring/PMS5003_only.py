# .......................... Connect to each Sensor of the Node ...........................
import csv
import board
import busio
import serial
import adafruit_bme280
import struct
#import os
import json
import datetime
import time
import smtplib
from subprocess import check_output
#import pandas as pd
import smtplib
import json
import datetime

def mail_alert1():                           
    fromaddr = email
    toaddrs  = email
    msg = 'Subject: {}\n\n{}'.format('Indoor Unit ' + sensorParameters['ID'] + ' ' +'PMS5003 error', 'Sensor' + '_' + sensorParameters['ID'] + ' PMS5003 error at ' + currentTime.strftime('%Y%m%d_%H%M%S'))

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
    msg = 'Subject: {}\n\n{}'.format('Indoor Unit ' + sensorParameters['ID'] + ' ' +'PMS5003 error', 'Sensor' + '_' + sensorParameters['ID'] + ' PMS5003 error at ' + currentTime.strftime('%Y%m%d_%H%M%S'))

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


#Once JSON file is created, open the file to read in sensorParameters
with open('/home/pi/SpokaneSchools/Cloud/sensorParameters.json') as json_file:
    sensorParameters=json.load(json_file)

# Create a unique filename for the current date.
currentHour = datetime.datetime.now().hour
currentTime = datetime.datetime.now()
currentDate = currentTime.date()
filename = sensorParameters['name'] + '_' + sensorParameters['ID'] + '_' +currentTime.strftime('%Y%m%d_%H%M%S') + '.csv'

### Initialize variables to store in CSV file.
data_file            = []

with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames = ["Datetime", "PM_0_3", "PM_0_5", 'PM_1', 'PM_2_5', 'PM_5', 'PM_10', 'PM1_standard', 'PM2_5_standard', 'PM10_standard', 'PM1_env', 'PM2_5_env', 'PM10_env'])
    writer.writeheader()
    #init_headers.to_csv(f, header=True)
    f.close()


#### Initialize Sensors
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=3000)
buffer = []

# .......................... Acquire and Store Sensor Data ...........................
while True:
    try:
    # If new day, then close current file and open a new file.
        if (datetime.datetime.now().date() != currentDate):
            with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'a') as f:
                wr = csv.writer(f, delimiter = ',')
                wr.writerows(data_file)
                f.close()
            currentHour = datetime.datetime.now().hour
            currentTime = datetime.datetime.now()
            currentDate = currentTime.date()
            data_file           = []
            filename = sensorParameters['name'] + '_' + sensorParameters['ID'] + '_' +currentTime.strftime('%Y%m%d_%H%M%S') + '.csv'
            with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'w') as f:
                writer = csv.DictWriter(f, fieldnames = ['Datetime', 'PM_0_3', 'PM_0_5', 'PM_1', 'PM_2_5', 'PM_5', 'PM_10', 'PM1_standard', 'PM2_5_standard', 'PM10_standard', 'PM1_env', 'PM2_5_env', 'PM10_env'])
                writer.writeheader()
                f.close()
            
        if datetime.datetime.now().hour != currentHour:
            with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'a') as f:
                #data_save.to_csv(f, header=False)
                wr = csv.writer(f, delimiter = ',')
                wr.writerows(data_file)
                #np.savetxt(f, data, delimiter = ',')
                f.close()
            currentHour = datetime.datetime.now().hour
            data_file           = []
        # Attempts to acquire and decode the data from the PMS5003 particulate matter sensor
        data = uart.read(32)  # read up to 32 bytes
        data = list(data)

        buffer += data
        while buffer and buffer[0] != 0x42:
            buffer.pop(0)

        if len(buffer) > 200:
            buffer = []  # avoid an overrun if all bad data
        if len(buffer) < 32:
            continue

        if buffer[1] != 0x4d:
            buffer.pop(0)
            continue

        frame_len = struct.unpack(">H", bytes(buffer[2:4]))[0]
        if frame_len != 28:
            buffer = []
            continue
        
        frame = struct.unpack(">HHHHHHHHHHHHHH", bytes(buffer[4:]))

        pm10_standard, pm25_standard, pm100_standard, pm10_env, \
            pm25_env, pm100_env, particles_03um, particles_05um, particles_10um, \
                particles_25um, particles_50um, particles_100um, skip, checksum = frame
        
        check = sum(buffer[0:30])
        
        if check != checksum:
            buffer = []
            continue

        # create error to see if email is sent (comment out after confirming function)
        #error = unknown_variable
        
        data_line = [datetime.datetime.now().isoformat(), particles_03um, particles_05um, particles_10um, particles_25um, particles_50um, particles_100um, pm10_standard, pm25_standard, pm100_standard, pm10_env, pm25_env, pm100_env]
        data_file.append(data_line)
        # print(type(data_line))
        # print(type(data))
        print(data_line)
        # print(data_file)
        print("Current time: ", datetime.datetime.now())

        buffer = buffer[32:]
    
        # Close JSON file
    
    except:
        
        currentTime = datetime.datetime.now()

        mail_alert1()
        #mail_alert2()
        time.sleep(3600)
