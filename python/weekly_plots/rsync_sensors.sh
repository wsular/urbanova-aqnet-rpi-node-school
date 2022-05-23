#!/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

rsync -avzhe ssh mattr@134.121.21.45:/mnt/data/vonw/fieldExperiments/2017_curr-Urbanova/ramboll /Users/matthew/work/PMS_5003_resample_backup/data/
