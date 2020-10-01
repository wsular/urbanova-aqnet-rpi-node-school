

import time
import json
import board
import busio
import adafruit_bme280
import csv
import datetime
import smtplib
import json
import datetime

def mail_alert1():                           
    fromaddr = email
    toaddrs  = email
    msg = 'Subject: {}\n\n{}'.format('Indoor Unit ' + sensorParameters['ID'] + ' ' +'BME280 error', 'Sensor' + '_' + sensorParameters['ID'] + ' BME280 error ' + currentTime.strftime('%Y%m%d_%H%M%S'))

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
    msg = 'Subject: {}\n\n{}'.format('Indoor Unit ' + sensorParameters['ID'] + ' ' +'BME280 error', 'Sensor' + '_' + sensorParameters['ID'] + ' BME280 error at ' + currentTime.strftime('%Y%m%d_%H%M%S'))

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
    


# Create library object using our Bus I2C port
i2c = busio.I2C(board.SCL, board.SDA)
bme280 = adafruit_bme280.Adafruit_BME280_I2C(i2c)


# change this to match the location's pressure (hPa) at sea level
bme280.sea_level_pressure = 1013.25

#Once JSON file is created, open the file to read in sensorParameters
with open('/home/pi/SpokaneSchools/Cloud/sensorParameters.json') as json_file:
    sensorParameters=json.load(json_file)

# Create a unique filename for the current date.
currentHour = datetime.datetime.now().hour
currentTime = datetime.datetime.now()
currentDate = currentTime.date()
filename = 'BME_' + sensorParameters['name'] + '_' + sensorParameters['ID'] + '_' +currentTime.strftime('%Y%m%d_%H%M%S') + '.csv'

### Initialize variables to store in CSV file.
data_file            = []


with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames = ["Datetime","temp", "P", "RH"])
    writer.writeheader()
    #init_headers.to_csv(f, header=True)
    f.close()


while True:
    try:
        if (datetime.datetime.now().date() != currentDate):
            with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'a') as f:
                wr = csv.writer(f, delimiter = ',')
                wr.writerows(data_file)
                f.close()
            currentHour = datetime.datetime.now().hour
            currentTime = datetime.datetime.now()
            currentDate = currentTime.date()
            data_file = []

            filename = 'BME_' + sensorParameters['name'] + '_' + sensorParameters['ID'] + '_' +currentTime.strftime('%Y%m%d_%H%M%S') + '.csv'
            with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'w') as f:
                writer = csv.DictWriter(f, fieldnames = ['Datetime', 'temp', 'P', 'RH'])
                writer.writeheader()
                f.close()

        if datetime.datetime.now().hour != currentHour:
            with open('/home/pi/SpokaneSchools/Data/Default_Frequency/' + filename, 'a') as f:
                wr = csv.writer(f, delimiter = ',')
                wr.writerows(data_file)
                f.close()
            currentHour = datetime.datetime.now().hour
            data_file = []
        
        # create error to see if email is sent (comment out after confirming function)
        #error = unknown_variable
        data_line = [datetime.datetime.now().isoformat(), bme280.temperature, bme280.pressure, bme280.humidity]
        data_file.append(data_line)
        print(data_line)
        print("\nTemperature: %0.1f C" % bme280.temperature)
        print("Humidity: %0.1f %%" % bme280.humidity)
        print("Pressure: %0.1f hPa" % bme280.pressure)
        print("Altitude = %0.2f meters" % bme280.altitude)
        time.sleep(120)
        
    except:
        currentTime = datetime.datetime.now()

        mail_alert1()
        #mail_alert2()
        time.sleep(3600)

