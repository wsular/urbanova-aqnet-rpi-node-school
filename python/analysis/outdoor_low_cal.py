#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 16:37:32 2021

@author: matthew
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 09:46:46 2021

@author: matthew
"""

# the adjustment in parentheses is from the Paccar roof test (calibrating the Clarity node to the Reference Clarity node), 
# after that is the adjustment determined from the SRCAA winter overlap of the Reference Clarity node to the BAM

# note that the in parentheses value is the "raw" measurement going into the MLR, although it has already obviously been calibrated from the Clarity node to Clarity Reference node at this point


def outdoor_cal_low(outdoor, name, time_period):
    
    if time_period =='6':
        outdoor_start_1 = '2019-09-01 00:00'
        outdoor_end_1 = '2020-09-10 23:00'
        
        outdoor_start_2 = '2020-09-21 20:00'
        outdoor_end_2 = '2021-02-21 00:00'   # for end of analysis period
      #  outdoor_end_2 = '2021-03-09 00:00'    # for end of data for sending to solmaz
        outdoor_1 = outdoor.loc[outdoor_start_1:outdoor_end_1]
        outdoor_2 = outdoor.loc[outdoor_start_2:outdoor_end_2]
        
        outdoor_cut = outdoor_1.append(outdoor_2)
        outdoor_cut = outdoor_cut.sort_index()
    
    
    elif time_period == '3':
    
        start_time = '2019-09-01 07:00'  #
        end_time = '2020-09-11 07:00'    #
        outdoor_cut = outdoor.copy()
        outdoor_cut = outdoor.loc[start_time:end_time]  
        #
    else:
        outdoor_cut = outdoor.copy()
    
    if name == 'Audubon': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.36)/1.0739)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242     
    
    else:
            pass
        
    if name == 'Adams': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+0.93)/1.1554)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242   
    
    else:
            pass

    if name == 'Balboa': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.2878)/1.2457)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242  
    
    
    else:
            pass      
        
    if name == 'Browne': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.4771)/1.1082)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242   
    
    
    else:
            pass 

    if name == 'Grant': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+1.0965)/1.29)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242  
    
    else:
            pass

    if name == 'Jefferson': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+0.7099)/1.1458)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242
    
    else:
            pass

    if name == 'Lidgerwood': 
        outdoor_cut['PM2_5_corrected'] = (outdoor_cut.PM2_5-1.1306)/0.98*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242 
    
    else:
            pass

    if name == 'Regal': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.247)/0.98)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242          
    
    else:
            pass

    if name == 'Sheridan': 
        outdoor_cut['PM2_5_corrected'] =  ((outdoor_cut.PM2_5+0.6958)/1.1468)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242 
    
    else:
            pass

    if name == 'Stevens': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+0.8901)/1.2767)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242  
    
    else:
            pass



    return outdoor_cut



























