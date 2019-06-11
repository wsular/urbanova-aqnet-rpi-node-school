
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

3. Install the Stretch operating system on the memory card using the instructions at (and then put in raspberry pi):
       https://howchoo.com/g/ndg2mtbmnmn/how-to-install-raspbian-stretch-on-the-raspberry-pi

4. Connect raspberry pi to monitor with ethernet, hdmi, mouse, keyboard, then power on 

5. On first boot (or In command terminal, type `sudo raspi-config`):   
       Using arrow keys to navigate
    1. NEED to RESIZE for SD card?
    2. Change the password. Use the lab password and appended X, where X is the sensor unit number.
    3. Set appropriate locale, keyboard, and timezone
    4. ~~Set the hostname to `airqualityX`~~, where X is the rooftop unit number. 
            Will set later using Pi's serial number (IS THIS NEEDED?)
    5. Enable SPI
    6. Enable I2C
    7. Disable shell on serial port, but keep the serial port hardware enabled.
    8. Enable SSH server
    9. Disable wait for network at boot (currently enabled on prototype, should it be?)
    10. Choose to "Update" in raspi-config
   
6.  Type sudo reboot if not rebooting from first time setup process

7. After booting raspi, change keyboard settings:

   * rasberry icon - preferencese - mouse and keyboard - keyboard - keyboard layout: set to microsoft office and english (us)
  
   In command terminal type  
8. `sudo apt-get dist-upgrade`

9.  Install basic utilities (are all needed?):   
    `sudo apt-get install tux`

    `sudo apt-get install htop`

    `sudo apt-get install build-essential`  (Might already be up-to-date.)

    `sudo apt-get install python-dev`

    `sudo apt-get install ntp`
    
    turn off default time keeping service   
    `systemctl stop systemd-timesyncd`


10.  Enable NPT stats: edit `/etc/ntp.conf` to uncomment line starting
    with `statsdir /var/log/ntpstats/`

11. Setup watchdog service (is this needed?)
    1. install watchdog timer using `sudo apt-get install watchdog`
    2. edit `/boot/config.txt` to contain `dtoverlay=watchdog=on`
       [ref](https://github.com/raspberrypi/linux/issues/1285#issuecomment-182264729)
    3. fixup the systemd service file [thanks to](https://kd8twg.net/2015/10/30/raspberry-pi-enabling-watchdog-on-raspbian-jessie/):
       edit `/lib/systemd/system/watchdog.service` to contain:

        ```
        [Install]
        WantedBy=multi-user.target
        ```

    4. edit `/etc/watchdog.conf` to contain
       [ref](https://blog.kmp.or.at/watchdog-for-raspberry-pi/)

        ```
        watchdog-device = /dev/watchdog
        watchdog-timeout = 10
        interval = 2
        max-load-1 = 24
        ```

    5. enable service and start it using sytemctl
        `sudo systemctl enable watchdog`
        `sudo systemctl start watchdog`
        
    6. finally, test it with a fork bomb: `:(){ :|:& };:`
       the Pi should return a PID number, then hang, then reboot

12. Enable persistent system logs (IS THIS NEEDED?): `sudo mkdir -p /var/log/journal`
    [ref](https://www.digitalocean.com/community/tutorials/how-to-use-journalctl-to-view-and-manipulate-systemd-logs)

13. Download supporting packages for various sensors (need to check which of these actually needed)
    1. install `python-pip` (should we use this if installing python3 for sensor?)
    2. `pip install spidev`
    3. `mkdir SpokaneSchools/software/`   
    4. `cd SpokaneSchools/software`
    5. `git clone https://github.com/adafruit/Adafruit_Python_GPIO`
    6. `git clone https://github.com/adafruit/Adafruit_Python_BME280`
    7. `cd Adafruit_Python_GPIO && sudo python setup.py install`
    8.  `cd ..`
    9.  `cd Adafruit_Python_BME280 && sudo python setup.py install`
    10. `cd ..`
    11. `cd py-opc && sudo python setup.py install`
    12. `sudo apt-get install python-pandas`

# Sensor Setups

Connect all the sensors directly to the Pi. 

## PMS 5003 setup  

Set up pre-reqs for PMS 5003 sensor:
    From https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi

    Note that steps 1-9 at the top of this link are already completed - start under 'Update Your Pi and Python'

`sudo apt-get update`

`sudo apt-get upgrade`

`sudo pip3 install --ugrade setuptools`

`sudo apt-get install python3-pip`

`pip3 install RPI.GPIO`

`pip3 install adafruit-blinka`

To check if circuit python installed: copy and paste 'Hello blinka' script from web page into new script and save as blinkatest.py in desired directory and then

From command line (while in same directory as file), run `python3 blinkatest.py`
   
   ### For wiring:   
   https://learn.adafruit.com/pm25-air-quality-sensor/circuitpython-code

   ### For code (same as wiring site):   
https://learn.adafruit.com/pm25-air-quality-sensor/circuitpython-code

Make nano file or other text editor, copy code and save as .py file in /home/pi/SpokaneSchools/software directory (or use git clone - code is from adafruit not gound on github?)

To edit in Nano, type `nano filename.py` in command line. Use arrow keys to navigate. Use control+o, then press ENTER to save over file. Use  control+x to quit and return to command line.

   - edit pms 5003 code in Nano as follows
   - add `import datetime` to top
   - change `uart = busio.UART(board.TX, board.RX, baudrate=9600)` to    
  `uart = serial.Serial("/dev/ttyS0", baudrate=9600, timeout = 3000)`
   - add `print(datetime.datetime.now())` below other print statements
   - run in command terminal using `python3 filename.py`   
  (make sure in correct directory)

## BME280 setup  
- for wiring:
  https://www.raspberrypi-spy.co.uk/2016/07/using-bme280-i2c-temperature-pressure-sensor-in-python/

sensor code should be in directory from git command above 12.5

- run using `python3 bme280.py`

## Connecting to Cloud
1. Download connection kit from Urbanova
2. Add install and cd changes to init.sh
3. Add both_sensors.py to init.sh file and comment out hello_sensor.py lines
4. `chmod +x init.sh`
5. `sudo ./init.sh`
6. change sleep to (60)