import sys
#print(sys.version)
import board
import busio
from digitalio import DigitalInOut, Direction
import serial
import json
import datetime
import time
import adafruit_bme280
import math

### Use on initial run to set up json file, then comment out and uncomment opening the old_data file below and updated data values
DateTime = []
T = []
H = []
P = []
PM_0_3 = []
PM_0_5 = []
PM_1 = []
PM_2_5 = []
PM_5 = []
PM_10 = []

#with open('Sensor_Data.json') as json_file:
#    old_data=json.load(json_file)

#print(old_data)

#DateTime = old_data['Datetime']
#T = old_data['Temp']
#H = old_data['H']
#P = old_data['P']
#PM_0_3 = old_data['PM_0_3']
#PM_0_5 = old_data['PM_0_5']
#PM_1 = old_data['PM_1']
#PM_2_5 = old_data['PM_2_5']
#PM_5 = old_data['PM_5']
#PM_10 = old_data['PM_10']

def default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()

try:
    import struct
except ImportError:
    import ustruct as struct
 
led = DigitalInOut(board.D13)
led.direction = Direction.OUTPUT

### check version of python being used
#import sys
#print(sys.version)
 
#### Create library object using our Bus I2C port
i2c = busio.I2C(board.SCL, board.SDA)
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)
uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout=3000)
buffer = []
### OR create library object using our Bus SPI port
#spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
#bme_cs = digitalio.DigitalInOut(board.D10)
#bme280 = adafruit_bme280.Adafruit_BME280_SPI(spi, bme_cs)
 
### change this to match the location's pressure (hPa) at sea level
bme280.sea_level_pressure = 1013.25
 

'''
 Copyright Urbanova 2019 | Licensed under the Apache License, Version 2.0 (the "License")
 This file is distributed on an "AS IS" BASIS,WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
 either express or implied. See the License for the specific language governing permissions
 and limitations under the License.
'''

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient
import logging
import time
import argparse
import json


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

loopCount = 0

while True:
    print("\nTemperature: %0.1f C" % bme280.temperature)
    print("Humidity: %0.1f %%" % bme280.humidity)
    print("Pressure: %0.1f hPa" % bme280.pressure)
    print("Altitude = %0.2f meters" % bme280.altitude)
    #b = 17.62
    #c = 243.12
    #gamma = (b * bme280.temperature /(c + bme280.temperature)) + math.log(bme280.humidity / 100.0)
    #dewpoint = (c * gamma) / (b - gamma)
    #print("Dewpoint = %0.1f C" % dewpoint)
    
    data = uart.read(32)  # read up to 32 bytes
    data = list(data)
    print("read: ", data)          # this is a bytearray type
 
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
 
    print("Concentration Units (standard)")
    print("---------------------------------------")
    print("PM 1.0: %d\tPM2.5: %d\tPM10: %d" %
          (pm10_standard, pm25_standard, pm100_standard))
    print("Concentration Units (environmental)")
    print("---------------------------------------")
    print("PM 1.0: %d\tPM2.5: %d\tPM10: %d" % (pm10_env, pm25_env, pm100_env))
    print("---------------------------------------")
    print("Particles > 0.3um / 0.1L air:", particles_03um)
    print("Particles > 0.5um / 0.1L air:", particles_05um)
    print("Particles > 1.0um / 0.1L air:", particles_10um)
    print("Particles > 2.5um / 0.1L air:", particles_25um)
    print("Particles > 5.0um / 0.1L air:", particles_50um)
    print("Particles > 10 um / 0.1L air:", particles_100um)
    print("---------------------------------------")
 
    #buffer = buffer[32:]
    #print("Buffer ", buffer)
    print(datetime.datetime.now())
    date_time = datetime.datetime.now()
    date_time = default(date_time)
    
    ### Print single values to be used in RPI monitor (data file updated every cycle)
    file2write=open('/home/pi/SpokaneSchools/software/temperature_data','w')
    file2write.write(str(bme280.temperature))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/humidity_data','w')
    file2write.write(str(bme280.humidity))
    file2write.close

    file2write=open('/home/pi/SpokaneSchools/software/pressure_data','w')
    file2write.write(str(bme280.pressure))
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
    
    ### append lists with new readings for saving all data to RPI
    DateTime.append(date_time)
    T.append(bme280.temperature)
    H.append(bme280.humidity)
    P.append(bme280.pressure)
    PM_0_3.append(particles_03um)
    PM_0_5.append(particles_05um)
    PM_1.append(particles_10um)
    PM_2_5.append(particles_25um)
    PM_5.append(particles_50um)
    PM_10.append(particles_100um)
    
    ### build new dictionary with updated values
    sensor_data = {'Datetime':DateTime,'Temp':T,'P':P,'H':H,'PM_0_3':PM_0_3,'PM_0_5':PM_0_5,'PM_1':PM_1,'PM_2_5':PM_2_5,'PM_5':PM_5,'PM_10':PM_10}

    ### update json file that stores all sensor data on RPI
    with open("Sensor_Data.json","w") as f:
        json.dump(sensor_data, f, indent = 2,sort_keys=True)

    #build dictionary to send single data packets up to cloud
    Cloud_data = {"datetime":date_time,"T":bme280.temperature,"H":bme280.humidity,"P":bme280.pressure,"PM_0_3":particles_03um,"PM_0_5":particles_05um,"PM_1":particles_10um,"PM_2_5":particles_25um,"PM_5":particles_50um,"PM_10":particles_100um}
    #time.sleep(2)

    ###Original
    #with open('Sensor_Data.json', 'a') as json_file:
    #    json.dump(sensor_data,json_file, indent = 4,sort_keys=True,default=str)
    
    ### Send single json data packet to cloud
    messageJson = json.dumps(Cloud_data) # convert to json
    ucIoTDeviceClient.publish(deviceId, messageJson, 1) 

    print('Published to %s: %s\n' % (deviceId, messageJson)) # print console
    loopCount += 1 # increment counter
    time.sleep(60) # delay one minute
    buffer = buffer[32:]
    print("Buffer ", buffer)
    
# Publish `Hello Sensor ${sensorID}`to Urbanova Cloud once per second
#loopCount = 0
#while True:
#  message = {} # init empty message obj
#  message['message'] = 'Hello Sensor ' + deviceId # add `message` element
#  message['sequence'] = loopCount # add `sequence` element
#  messageJson = json.dumps(message) # convert to json
#  ucIoTDeviceClient.publish(deviceId, messageJson, 1) # publish to urbanova cloud
#  print('Published to %s: %s\n' % (deviceId, messageJson)) # print console
#  loopCount += 1 # increment counter
#  time.sleep(1) # delay one second