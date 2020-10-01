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
#DateTime       = []
#PM_0_3         = []
#PM_0_5         = []
#PM_1           = []
#PM_2_5         = []
#PM_5           = []
#PM_10          = []
#PM1_standard   = []
#PM2_5_standard = []
#PM10_standard  = []
#PM1_env        = []
#PM2_5_env      = []
#PM10_env       = []

#init_headers = pd.DataFrame(
#                {'DateTime': DateTime,
#                'PM_0_3':    PM_0_3,0
#                'PM_0_5':    PM_0_5,
#                'PM_1':      PM_1,
#                'PM_2_5':    PM_2_5,
#                'PM_5':      PM_5,
#                'PM_10':     PM_10,
#                'PM1_standard': PM1_standard,
#                'PM2_5_standard': PM2_5_standard,
#                'PM10_standard':  PM10_standard,
#                'PM1_env':  PM1_env,
#                'PM2_5_env': PM2_5_env,
#                'PM10_env': PM10_env
#                })
with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames = ["Datetime", "PM_0_3", "PM_0_5", 'PM_1', 'PM_2_5', 'PM_5', 'PM_10', 'PM1_standard', 'PM2_5_standard', 'PM10_standard', 'PM1_env', 'PM2_5_env', 'PM10_env'])
    writer.writeheader()
    #init_headers.to_csv(f, header=True)
    f.close()


#### Initialize Sensors
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=3000)
buffer = []

#i2c = busio.I2C(board.SCL, board.SDA)
#bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
#bme280.sea_level_pressure = 1013.25# Set this to the location's approximate pressure (hPa) at sea level (This is needed if we ever want to use bme280.altitude.)

# .......................... Acquire and Store Sensor Data ...........................
while True:
    # If new day, then close current file and open a new file.
    if (datetime.datetime.now().date() != currentDate):
       # data_save = pd.DataFrame(
       # {'DateTime': DateTime,
       # 'PM_0_3':   PM_0_3,
       # 'PM_0_5':   PM_0_5,
       # 'PM_1':     PM_1,
       # 'PM_2_5':   PM_2_5,
       # 'PM_5':     PM_5,
       # 'PM_10':    PM_10,
       # 'PM1_standard': PM1_standard,
       # 'PM2_5_standard': PM2_5_standard,
       # 'PM10_standard': PM10_standard,
       # 'PM1_env':  PM1_env,
       # 'PM2_5_env':PM2_5_env,
       # 'PM10_env': PM10_env
       # })
        #with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'w') as f:
        #    writer = csv.DictWriter(f, fieldnames = [' ', 'DateTime', 'PM_0_3', 'PM_0_5', 'PM_1', 'PM_2_5', 'PM_5', 'PM_10', 'PM1_standard', 'PM2_5_standard', 'PM10_standard', 'PM1_env', 'PM2_5_env', 'PM10_env'])
        #    writer.writeheader()
        #    f.close()
        with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'a') as f:
            wr = csv.writer(f, delimiter = ',')
            wr.writerows(data_file)
            #data_save.to_csv(f, header=False)
            #np.savetxt(f, data, delimiter = ',')
            f.close()
        currentHour = datetime.datetime.now().hour
        currentTime = datetime.datetime.now()
        currentDate = currentTime.date()
        #currentDate = datetime.datetime.now().date()
        #DateTime       = []
        data_file           = []
        #PM_0_3         = []
        #PM_0_5         = []
        #PM_1           = []
        #PM_2_5         = []
        #PM_5           = []
        #PM_10          = []
        #PM1_standard   = []
        #PM2_5_standard = []
        #PM10_standard  = []
        #PM1_env        = []
        #PM2_5_env      = []
        #PM10_env       = []
        filename = sensorParameters['name'] + '_' + sensorParameters['ID'] + '_' +currentTime.strftime('%Y%m%d_%H%M%S') + '.csv'
        with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames = ['Datetime', 'PM_0_3', 'PM_0_5', 'PM_1', 'PM_2_5', 'PM_5', 'PM_10', 'PM1_standard', 'PM2_5_standard', 'PM10_standard', 'PM1_env', 'PM2_5_env', 'PM10_env'])
            writer.writeheader()
            f.close()
    if datetime.datetime.now().hour != currentHour:
        #data_save = pd.DataFrame(
        #{'Datetime': DateTime,
        #'PM_0_3':    PM_0_3,
        #'PM_0_5':    PM_0_5,
        #'PM_1':      PM_1,
        #'PM_2_5':    PM_2_5,
        #'PM_5':      PM_5,
        #'PM_10':     PM_10,
        #'PM1_standard': PM1_standard,
        #'PM2_5_standard': PM2_5_standard,
        #'PM10_standard': PM10_standard,
        #'PM1_env': PM1_env,
        #'PM2_5_env': PM2_5_env,
        #'PM10_env':  PM10_env
        #})
        with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'a') as f:
            #data_save.to_csv(f, header=False)
            wr = csv.writer(f, delimiter = ',')
            wr.writerows(data_file)
            #np.savetxt(f, data, delimiter = ',')
            f.close()
        currentHour = datetime.datetime.now().hour
        data_file           = []
        #DateTime       = []
        #PM_0_3         = []
        #PM_0_5         = []
        #PM_1           = []
        #PM_2_5         = []
        #PM_5           = []
        #PM_10          = []
        #PM1_standard   = []
        #PM2_5_standard = []
        #PM10_standard  = []
        #PM1_env        = []
        #PM2_5_env      = []
        #PM10_env       = []
# Attempts to acquire and decode the data from the PMS5003 particulate matter sensor
    data = uart.read(32)  # read up to 32 bytes
    data = list(data)
    # print("read: ", data)          # this is a bytearray type

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

    # Stores the current time and data
    #DateTime.append(datetime.datetime.now().isoformat())
    #PM_0_3.append(particles_03um)
    #PM_0_5.append(particles_05um)
    #PM_1.append(particles_10um)
    #PM_2_5.append(particles_25um)
    #PM_5.append(particles_50um)
    #PM_10.append(particles_100um)
    #PM1_standard.append(pm10_standard)
    #PM2_5_standard.append(pm25_standard)
    #PM10_standard.append(pm100_standard)
    #PM1_env.append(pm10_env)
    #PM2_5_env.append(pm25_env)
    #PM10_env.append(pm100_env)
   # print(datetime.datetime.now())
   # print(particles_03um)
   # print(particles_05um)
    data_line = [datetime.datetime.now().isoformat(), particles_03um, particles_05um, particles_10um, particles_25um, particles_50um, particles_100um, pm10_standard, pm25_standard, pm100_standard, pm10_env, pm25_env, pm100_env]
    data_file.append(data_line)
   # print(type(data_line))
   # print(type(data))
    print(data_line)
   # print(data_file)
    print("Current time: ", datetime.datetime.now())

 #   print("---------------------------------------")
 #   print("Particles > 0.3um / 0.1L air:", PM_0_3[-1])
 #   print("Particles > 0.5um / 0.1L air:", PM_0_5[-1])
 #   print("Particles > 1.0um / 0.1L air:", PM_1[-1])
 #   print("Particles > 2.5um / 0.1L air:", PM_2_5[-1])
 #   print("Particles > 5.0um / 0.1L air:", PM_5[-1])
 #   print("Particles > 10 um / 0.1L air:", PM_10[-1])
 #   print("---------------------------------------")

    # Store all sensor data on RPI in JSON file
   # sensor_data = {'name':           sensorParameters['name'],
  #                 'ID':             sensorParameters['ID'],
 #                  'Type':           sensorParameters['Type'],
 #                  'description':    sensorParameters['description'],
 #                  'contact':        sensorParameters['contact'],
 #                  'Datetime':       DateTime,
 #                  'PM_0_3':         PM_0_3,
 #                  'PM_0_5':         PM_0_5,
 #                  'PM_1':           PM_1,
  #                 'PM_2_5':         PM_2_5,
  #                 'PM_5':           PM_5,
  #                 'PM_10':          PM_10,
 #                  'PM1_standard':   PM1_standard,
  # #                'PM2_5_standard': PM2_5_standard,
  #                 'PM10_standard':  PM10_standard,
  #                 'PM1_env':        PM1_env,
  #                 'PM2_5_env':      PM2_5_env,
  #                 'PM10_env':       PM10_env
  #                 }
   # json.dump(sensor_data, json_file, indent = 2,sort_keys=True)
        
    # Reset data buffer for PMS5003
    buffer = buffer[32:]
    
    # Close JSON file
    

