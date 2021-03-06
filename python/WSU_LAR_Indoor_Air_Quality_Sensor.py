'''
This Python script acquires and stores data from the WSU LAR indoor air quality sensors
that were built for the Ramboll project. This project installed both indoor and outdoor
sensors in elementary schools in Spokane, Washington.

Contact: Von P. Walden, Washington State University
Date:    16 July 2019
'''

def local_wifi():
    local_wifi_arr = check_output("ifconfig").decode('utf-8')
    connection_test = local_wifi_arr.rfind("inet .....")    # input target IP range here
    if connection_test > 0:
        return True
    else:
        return False

def mail_alert(sensor):                           # input sensor type of bad data
    fromaddr = email
    toaddrs  = email
    msg = 'Subject: {}\n\n{}'.format('BAD DATA FROM INDOOR AQ SENSOR', 'Erroneous data record from' + '_' + sensor + '_' + currentTime.strftime('%Y%m%d_%H%M%S') + '_' + 'Sensor' + '_' + sensorParameters['ID'])

# Credentials (if needed)
    username = email
    password = pw

# The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()

def mail_alert2(sensor):                           # input sensor type of bad data
    fromaddr = email
    toaddrs  = email
    msg = 'Subject: {}\n\n{}'.format('Publishing Error', 'Publishing Error from' + '_' + sensor + '_' + currentTime.strftime('%Y%m%d_%H%M%S') + '_' + 'Sensor' + '_' + sensorParameters['ID'])

# Credentials (if needed)
    username = email
    password = pw

# The actual mail send
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()


def acquireBME280():
    T.append(bme280.temperature)
    RH.append(bme280.humidity)
    P.append(bme280.pressure)
    return

def writeRPiMonitor():
    '''
    Print single values to files to be used in RPi monitor; 
    data files are updated every cycle.

    Written by: Matthew Roetcisoender
    Created on: June 2019
    '''
    file2write=open('/home/pi/SpokaneSchools/software/temperature_data','w')
    file2write.write(str(T[-1]))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/humidity_data','w')
    file2write.write(str(RH[-1]))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/pressure_data','w')
    file2write.write(str(P[-1]))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/PM_0_3_data','w')
    file2write.write(str(particles_03um))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/PM_0_5_data','w')
    file2write.write(str(particles_05um))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/PM_1_data','w')
    file2write.write(str(particles_10um))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/PM_2_5_data','w')
    file2write.write(str(particles_25um))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/PM_5_data','w')
    file2write.write(str(particles_50um))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/PM_10_data','w')
    file2write.write(str(particles_100um))
    file2write.close
    
    return

#.............................. Make sure connected to internet before start...............
import datetime
import time
from subprocess import check_output
   
currentTime = datetime.datetime.now()

local_wifi()
    
while not local_wifi():
    print('no wifi connection')
    with open("/home/pi/SpokaneSchools/Data/wifi_errors/wifi_errors.txt", "a") as myfile:
        myfile.write('no_wifi' + '_' + currentTime.strftime('%Y%m%d_%H%M%S') + "\n")
        myfile.close
    time.sleep(120)

# ............................. Connect to the Urbanova Cloud .............................
'''
 Copyright Urbanova 2019 | Licensed under the Apache License, Version 2.0 (the "License")
 This file is distributed on an "AS IS" BASIS,WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 either express or implied. See the License for the specific language governing permissions
 and limitations under the License.
'''
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import argparse
import logging
import time

### Custom MQTT message callback
def customCallback(client, userdata, message):
    print("Received a new message: ")
    print(message.payload)
    print("from topic: ")
    print(message.topic)
    print("--------------\n\n")

### Read in command-line parameters
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--rootCA", action="store", required=True, dest="rootCAPath", help="Root CA file path")
parser.add_argument("-c", "--cert", action="store", required=True, dest="certificatePath", help="Certificate file path")
parser.add_argument("-k", "--key", action="store", required=True, dest="privateKeyPath", help="Private key file path")
parser.add_argument("-d", "--device", action="store", required=True, dest="deviceId", help="Device Identifier")

### Urbanova Cloud IoT Custom Endpoint / MQTT Broker hosted at AWS
ucIoTCustomEndpoint = "a1siobcc26zf4j-ats.iot.us-west-2.amazonaws.com"

### Parse Arguments
args = parser.parse_args()
rootCAPath = args.rootCAPath # rootCA path
certificatePath = args.certificatePath # thing certifiate path
privateKeyPath = args.privateKeyPath # thing private key path
deviceId = args.deviceId # thing deviceId (autogenerated at time of Urbanova Cloud IoT Data Source creation)


### Configure logging
logger = logging.getLogger("AWSIoTPythonSDK.core")
logger.setLevel(logging.DEBUG)
f_handler = logging.FileHandler("/home/pi/SpokaneSchools/Data/logging/logs.log")
f_handler.setLevel(logging.DEBUG)
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)


### Init Urbanova Cloud IoT MQTT Client using TLSv1.2 Mutual Authentication
ucIoTDeviceClient = None  # initialize var
ucIoTDeviceClient = AWSIoTMQTTClient(deviceId) # The client class that connects to and accesses AWS IoT over MQTT v3.1/3.1.1.
ucIoTDeviceClient.configureEndpoint(ucIoTCustomEndpoint, 8883) # MQTT Broker host address and default port (TLS)
ucIoTDeviceClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath) # certs and key


### Configure Urbanova Cloud IoT Device Client Connection Settings (reference: https://s3.amazonaws.com/aws-iot-device-sdk-python-docs/sphinx/html/index.html)
ucIoTDeviceClient.configureAutoReconnectBackoffTime(1, 32, 20)
ucIoTDeviceClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publish queueing
ucIoTDeviceClient.configureDrainingFrequency(2)  # Draining: 2 Hz
ucIoTDeviceClient.configureConnectDisconnectTimeout(10)  # 10 sec
ucIoTDeviceClient.configureMQTTOperationTimeout(5)  # 5 sec


### Connect to Urbanova Cloud IoT
ucIoTDeviceClient.connect()
time.sleep(2)
# .........................................................................................

# .......................... Connect to each Sensor of the Node ...........................

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
#Create JSON file for sensorParameters (just update ID for each sensor)

#name='WSU_LAR_Indoor_Air_Quality_Node'
#ID='PT'
#Type='WSU LAR Indoor Air Quality Node'
#description='Indoor air quality sensor package (node) built at Washington State University Laboratory for Atmospheric Research. Sensors include PMS5003 (particulate matter) and BME280 (TPU).'
#contact='Von P. Walden, Washington State University, v.walden@wsu.edu'
#timeInterval=120

#sensorParameters={'name':name,'ID':ID,'Type':Type,'description':description,'contact':contact,'timeInterval':timeInterval}

#with open("sensorParameters.json","w") as f:
#        json.dump(sensorParameters, f, indent = 2,sort_keys=True)

#Once JSON file is created, open the file to read in sensorParameters
with open('sensorParameters.json') as json_file:
    sensorParameters=json.load(json_file)

# Create a unique filename for the current date.
currentTime = datetime.datetime.now()
currentDate = currentTime.date()
#currentDate = datetime.datetime.now().date()
filename = sensorParameters['name'] + '_' + sensorParameters['ID'] + '_' +currentTime.strftime('%Y%m%d_%H%M%S') + '.json'

### Initialize variables to store in JSON file.
DateTime       = []
T              = []
RH             = []
P              = []
PM_0_3         = []
PM_0_5         = []
PM_1           = []
PM_2_5         = []
PM_5           = []
PM_10          = []
PM1_standard   = []
PM2_5_standard = []
PM10_standard  = []
PM1_env        = []
PM2_5_env      = []
PM10_env       = []

#### Initialize Sensors
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=3000)
buffer = []

i2c = busio.I2C(board.SCL, board.SDA)
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
bme280.sea_level_pressure = 1013.25# Set this to the location's approximate pressure (hPa) at sea level (This is needed if we ever want to use bme280.altitude.)

with open('/home/pi/SpokaneSchools/software/Name_1.txt','r') as file:
    email=file.read()

with open('/home/pi/SpokaneSchools/software/Name_2.txt','r') as file:
    pw=file.read()

# .......................... Acquire and Store Sensor Data ...........................
while True:
    # If new day, then close current JSON file and open a new file.
    if (datetime.datetime.now().date() != currentDate):
        currentTime = datetime.datetime.now()
        currentDate = currentTime.date()
        #currentDate = datetime.datetime.now().date()
        DateTime       = []
        T              = []
        RH             = []
        P              = []
        PM_0_3         = []
        PM_0_5         = []
        PM_1           = []
        PM_2_5         = []
        PM_5           = []
        PM_10          = []
        PM1_standard   = []
        PM2_5_standard = []
        PM10_standard  = []
        PM1_env        = []
        PM2_5_env      = []
        PM10_env       = []
        filename = sensorParameters['name'] + '_' + sensorParameters['ID'] + '_' +currentTime.strftime('%Y%m%d_%H%M%S') + '.json'
    json_file = open('/home/pi/SpokaneSchools/Data/Good_Data/' + filename, 'w')

    try:  # Attempts to acquire and decode the data from the PMS5003 particulate matter sensor
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
        DateTime.append(datetime.datetime.now().isoformat())
        PM_0_3.append(particles_03um)
        PM_0_5.append(particles_05um)
        PM_1.append(particles_10um)
        PM_2_5.append(particles_25um)
        PM_5.append(particles_50um)
        PM_10.append(particles_100um)
        PM1_standard.append(pm10_standard)
        PM2_5_standard.append(pm25_standard)
        PM10_standard.append(pm100_standard)
        PM1_env.append(pm10_env)
        PM2_5_env.append(pm25_env)
        PM10_env.append(pm100_env)

    except:
        print('!! Erroneous data record from PMS5003 !!')
        print('    Skipping measurement and trying again...')
        #os.system('echo "ALERT: Problem with WSU LAR Indoor AQ sensor (PMS5003 Data)" | mail -s "ALERT:  Problem with WSU LAR Indoor AQ sensor on ' + datetime.datetime.utcnow().strftime('%Y%m%d %H') + ':00 UTC" v.walden@wsu.edu')
        #os.system('echo "ALERT: Problem with WSU LAR Indoor AQ sensor (PMS5003 Data)" | mail -s "ALERT:  Problem with WSU LAR Indoor AQ sensor on ' + datetime.datetime.utcnow().strftime('%Y%m%d %H') + ':00 UTC" matthew.s.roetcisoe@wsu.edu')
        mail_alert('PMS_5003')
        with open("/home/pi/SpokaneSchools/Data/PMS5003_errors/errors_PMS_5003.txt", "a") as myfile:
            myfile.write('Erroneous data record from PMS5003' + currentTime.strftime('%Y%m%d_%H%M%S') + '_' + 'Sensor' + '_' + sensorParameters['ID'] + "\n")
            myfile.close
        print(buffer)
        #PMS_5003_startup()
        uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=3000)
        buffer = []
        time.sleep(10)
        continue
    
    try:  # Attempts to acquire and decode the data from the BME280 meteorlogical sensor
        acquireBME280()
    except:
        print('!! Erroneous data record from BME280 !!')
        print('    Skipping measurement and trying again...')
        #os.system('echo "ALERT: Problem with WSU LAR Indoor AQ sensor (BME280 Data)" | mail -s "ALERT:  Problem with WSU LAR Indoor AQ sensor on ' + datetime.datetime.utcnow().strftime('%Y%m%d %H') + ':00 UTC" v.walden@wsu.edu')
        #os.system('echo "ALERT: Problem with WSU LAR Indoor AQ sensor (BME280 Data)" | mail -s "ALERT:  Problem with WSU LAR Indoor AQ sensor on ' + datetime.datetime.utcnow().strftime('%Y%m%d %H') + ':00 UTC" matthew.s.roetcisoe@wsu.edu')
        mail_alert('BME_280')
        with open("/home/pi/SpokaneSchools/Data/BME280_errors/errors_PMS_5003.txt", "a") as myfile:
            myfile.write('Erroneous data record from BME280' + currentTime.strftime('%Y%m%d_%H%M%S') + '_' + 'Sensor' + '_' + sensorParameters['ID'] + "\n")
            myfile.close
        print(buffer)
        i2c = busio.I2C(board.SCL, board.SDA)
        bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
        bme280.sea_level_pressure = 1013.25
        #BME_280_startup()
        time.sleep(10)
        continue
    
    print('Current time: ', DateTime[-1])

    print("Temperature       = %0.1f C" % T[-1])
    print("Relative Humidity = %0.1f percent" % RH[-1])
    print("Pressure          = %0.1f hPa" % P[-1])

    #print("Concentration Units (standard)")
    #print("---------------------------------------")
    #print("PM 1.0: %d\tPM2.5: %d\tPM10: %d" % (pm10_standard, pm25_standard, pm100_standard))
    #print("Concentration Units (environmental)")
    #print("---------------------------------------")
    #print("PM 1.0: %d\tPM2.5: %d\tPM10: %d" % (pm10_env, pm25_env, pm100_env))
    print("---------------------------------------")
    print("Particles > 0.3um / 0.1L air:", PM_0_3[-1])
    print("Particles > 0.5um / 0.1L air:", PM_0_5[-1])
    print("Particles > 1.0um / 0.1L air:", PM_1[-1])
    print("Particles > 2.5um / 0.1L air:", PM_2_5[-1])
    print("Particles > 5.0um / 0.1L air:", PM_5[-1])
    print("Particles > 10 um / 0.1L air:", PM_10[-1])
    print("---------------------------------------")

    writeRPiMonitor()

    # Store all sensor data on RPI in JSON file
    sensor_data = {'name':           sensorParameters['name'],
                   'ID':             sensorParameters['ID'],
                   'Type':           sensorParameters['Type'],
                   'description':    sensorParameters['description'],
                   'contact':        sensorParameters['contact'],
                   'timeInterval':   sensorParameters['timeInterval'],
                   'latitude':       sensorParameters['latitude'],
                   'longitude':      sensorParameters['longitude'],
                   'Datetime':       DateTime,
                   'Temp':           T,
                   'P':              P,
                   'RH':             RH,
                   'PM_0_3':         PM_0_3,
                   'PM_0_5':         PM_0_5,
                   'PM_1':           PM_1,
                   'PM_2_5':         PM_2_5,
                   'PM_5':           PM_5,
                   'PM_10':          PM_10,
                   'PM1_standard':   PM1_standard,
                   'PM2_5_standard': PM2_5_standard,
                   'PM10_standard':  PM10_standard,
                   'PM1_env':        PM1_env,
                   'PM2_5_env':      PM2_5_env,
                   'PM10_env':       PM10_env
                   }
    json.dump(sensor_data, json_file, indent = 2,sort_keys=True)

    ### Send single json data packet to cloud
    Cloud_data = {'name':           sensorParameters['name'],
                  'ID':             sensorParameters['ID'],
                  'Type':           sensorParameters['Type'],
                  'description':    sensorParameters['description'],
                  'contact':        sensorParameters['contact'],
                  'timeInterval':   sensorParameters['timeInterval'],
                  'latitude':       sensorParameters['latitude'],
                  'longitude':      sensorParameters['longitude'],
                  "datetime":       DateTime[-1],
                  "T":              T[-1],
                  "RH":             RH[-1],
                  "P":              P[-1],
                  "PM_0_3":         PM_0_3[-1],
                  "PM_0_5":         PM_0_5[-1],
                  "PM_1":           PM_1[-1],
                  "PM_2_5":         PM_2_5[-1],
                  "PM_5":           PM_5[-1],
                  "PM_10":          PM_10[-1],
                  'PM1_standard':   PM1_standard[-1],
                  'PM2_5_standard': PM2_5_standard[-1],
                  'PM10_standard':  PM10_standard[-1],
                  'PM1_env':        PM1_env[-1],
                  'PM2_5_env':      PM2_5_env[-1],
                  'PM10_env':       PM10_env[-1]
                  }
    
    messageJson = json.dumps(Cloud_data) # convert to json
    
    local_wifi()
    
    while not local_wifi():
        print('no wifi connection')
        with open("/home/pi/SpokaneSchools/Data/wifi_errors/wifi_errors.txt", "a") as myfile:
            myfile.write('no_wifi' + '_' + currentTime.strftime('%Y%m%d_%H%M%S') + '_' + 'Sensor' + '_' + sensorParameters['ID'] + "\n")
            myfile.close
        time.sleep(120)
    try:
        ucIoTDeviceClient.publish(deviceId, messageJson, 1) 
        print('Published to %s: %s\n' % (deviceId, messageJson)) # print console
    except:
        logger.debug("Error in Publishing", exc_info=True)
        time.sleep(600)
        print('1')
        mail_alert2('Indoor Unit')
        print('2')
        time.sleep(600)
    
    # Reset data buffer for PMS5003
    buffer = buffer[32:]
    
    # Close JSON file
    json_file.close()
    
    # Waits for desired time interval
    time.sleep(sensorParameters['timeInterval'])
