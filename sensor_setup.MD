
# Urbanova Air Quality  - Indoor School Units

- Author: Matthew Roetcisoender

## Components

* Single-board computer (Raspberry Pi 3 - Model B+)
    <https://www.adafruit.com/product/3775>
* Temp,Pressure,Humidity sensor (BME280; Adafruit)
    <https://www.adafruit.com/product/2652>
* Particulate sensor (PMS 5003; Adafruit)
    <https://www.adafruit.com/product/3686>

### Initial Raspberry PiSetup

1. Purchase 16-GB SD card (GET EXACT MODEL NEEDED INPUT HERE.)

2. Download the last version of the Stretch operating system onto SD card (choose option with desktop and recommended software) for the Raspberry Pi computer at:
       https://www.raspberrypi.org/downloads/raspbian/

3. Install the Stretch operating system on the memory card using the instructions at (and then put sd card in raspberry pi):
       https://howchoo.com/g/ndg2mtbmnmn/how-to-install-raspbian-stretch-on-the-raspberry-pi

4. Connect raspberry pi to monitor with ethernet, hdmi, mouse, keyboard, then power on 

5. On first boot (or In command terminal, type `sudo raspi-config`):   
       Using arrow keys to navigate
    1. NEED to RESIZE for SD card? - no
    2. Change the password. Use the lab password and appended X, where X is the sensor unit number.
    3. Set appropriate locale, keyboard, and timezone
    4. ~~Set the hostname to `airqualityX`~~, where X is the rooftop unit number - skip  
            Will set later using Pi's serial number (IS THIS NEEDED?)
    5. Select interfacing options
       1. Enable SPI
       2. Enable I2C
       3. Disable shell on serial port, but keep the serial port hardware enabled.
       4. Enable SSH server
    6. In boot options - disable wait for network at boot (currently enabled on prototype, should it be?)
    7.  Choose to "Update" in raspi-config
   
6.  Type sudo reboot if not rebooting from first time setup process

7. After booting raspi, change keyboard settings:

   * rasberry icon - preferencese - mouse and keyboard - keyboard - keyboard layout: set to microsoft office and english (us)
  
   In command terminal home directory type  
8. `sudo apt-get dist-upgrade`

9.  Install basic utilities:   
    `sudo apt-get install tux` - not found 

    `sudo apt-get install htop` (Might already be up-to-date)

    `sudo apt-get install build-essential`  (Might already be up-to-date.)

    `sudo apt-get install python-dev`

    `sudo apt-get install ntp`
    
    Turn off default time keeping service   
    `systemctl stop systemd-timesyncd`


10.  Enable NPT stats: edit `/etc/ntp.conf` to uncomment line starting
    with `statsdir /var/log/ntpstats/`   
    To do so: `cd /etc`   
    Then: `sudo nano ntp.conf`   
    Then: uncomment line
    Save with: ctrl+o (and press enter)   
    Exit with: ctrl+x


11. Setup watchdog service
    1. install watchdog timer using `sudo apt-get install watchdog`
    2. edit `/boot/config.txt` to add `dtoverlay=watchdog=on` to bottom of file
       [ref](https://github.com/raspberrypi/linux/issues/1285#issuecomment-182264729)   
       To do so: `cd /boot`   
       To edit: ` sudo nano config.txt` make change then save and exit as in 10
    3. fixup the systemd service file [thanks to](https://kd8twg.net/2015/10/30/raspberry-pi-enabling-watchdog-on-raspbian-jessie/):
       edit `/lib/systemd/system/watchdog.service` to contain:

        ```
        [Install]
        WantedBy=multi-user.target
        (note that you just cange the 'default' part of this line to 'multi-user')
        ```

    4. edit `/etc/watchdog.conf` to contain
       [ref](https://blog.kmp.or.at/watchdog-for-raspberry-pi/)

        ```
        watchdog-device = /dev/watchdog
        watchdog-timeout = 10
        interval = 2
        max-load-1 = 24
        ```

    5. return to home directory and enable service and start it using sytemctl   
        `sudo systemctl enable watchdog`   
        `sudo systemctl start watchdog`
        
    6. finally, test it with a fork bomb: `:(){ :|:& };:`
       the Pi should return a PID number, then hang 10 sec, then reboot

12. Enable persistent system logs (IS THIS NEEDED?): `sudo mkdir -p /var/log/journal`
    [ref](https://www.digitalocean.com/community/tutorials/how-to-use-journalctl-to-view-and-manipulate-systemd-logs)

13. Go to home directory and download supporting packages for various sensors (need to check which of these actually needed)
    1. install `python-pip` (should we use this if installing python3 for sensor?)
    2. `pip install spidev`
    3. `mkdir SpokaneSchools`   
    4. `cd ./SpokaneSchools`
    5. `mkdir software`
    6. `cd ./software`
    7. `git clone https://github.com/adafruit/Adafruit_Python_GPIO` 
    8. `git clone https://github.com/adafruit/Adafruit_Python_BME280`
    9. `cd ./Adafruit_Python_GPIO && sudo python setup.py install`
    10. `cd ..`
    11. `cd ./Adafruit_Python_BME280 && sudo python setup.py install`
    12. `cd ..`
    13. `cd ..`
    14. `mkdir Cloud`
    15. `cd py-opc && sudo python setup.py install` - not needed
    16. `sudo apt-get install python-pandas` -not needed

# Sensor Setups

Connect all the sensors directly to the Pi. 

## PMS 5003 setup  

Set up pre-reqs for PMS 5003 sensor:
    From https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi

    Note that steps 1-9 at the top of this link are already completed - start under 'Update Your Pi and Python'
start in home directory   
`sudo apt-get update`

`sudo apt-get upgrade`

`sudo pip3 install --ugrade setuptools`

`sudo apt-get install python3-pip`

`pip3 install RPI.GPIO`

`pip3 install adafruit-blinka`   

`sudo pip3 install adafruit-circuitpython-bme280`

To check if circuit python installed: copy and paste 'Hello blinka' script from web page into new script and save as blinkatest.py in desired directory and then

From command line (while in same directory as file), run `python3 blinkatest.py`
   
   ### For wiring:   
   https://learn.adafruit.com/pm25-air-quality-sensor/circuitpython-code

   ### For code (same as wiring site):   
https://learn.adafruit.com/pm25-air-quality-sensor/circuitpython-code

Make nano file or other text editor, copy code and save as sensor.py file in /home/pi/SpokaneSchools/software directory 

Edit sensor.py in Nano, type `nano sensor.py` in command line while in ./software directory. Use arrow keys to navigate. Use control+o, then press ENTER to save over file. Use  control+x to quit and return to command line.

   - edit sensor.py  code in Nano as follows
   - add `import datetime` to top
   - add `import serial` to top
   - change `uart = busio.UART(board.TX, board.RX, baudrate=9600)` to    
  `uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout = 3000)`
   - add `print(datetime.datetime.now())` below other print statements
   - run in command terminal using `python3 sensor.py`   
  (./software directory)

## BME280 setup  
- for wiring:
  https://www.raspberrypi-spy.co.uk/2016/07/using-bme280-i2c-temperature-pressure-sensor-in-python/

- for code (scroll to bottom and copy last code section):
  https://learn.adafruit.com/adafruit-bme280-humidity-barometric-pressure-temperature-sensor-breakout/python-circuitpython-test
- type `nano` while in ./software
- copy bme280 code into the nano file and save (name as bme280.py)

- run using `python3 bme280.py`

## Connecting to Cloud
1. Download connection kit from Urbanova and extract to /home/pi/SpokaneSchools/Cloud
2. Edit init.sh in the /home/pi/SpokaneSchools/Cloud directory
   1. change `pushd` and `popd` to `cd's`
   2. add `pip3 install RPI.GPIO`
   3. add `pip3 install adafruit-blinka`
   4. add `pip3 install AWSIotPythonSDK`
   5. copy and paste last two lines, comment out `hello_sensor.py` lines and change the python command to `python3 /home/pi/SpokaneSchools/Cloud/both_sensors.py`
3. use flash drive to copy both_sensors.py to /home/pi/SpokaneSchools/Cloud
4. edit /home/pi/SpokaneSchools/both_sensors.py
   1. change `sleep(1)` to `sleep(60)`
5. in terminal while in /home/pi/SpokaneSchools/Cloud type `chmod +x init.sh`
6. `sudo ./init.sh`
