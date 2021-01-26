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

# the adjustment in paranethese is to the Paccar roof test, after that is the adjustment determined from the SRCAA winter overlap


def outdoor_cal_low(outdoor, name):
    
    start_time = '2019-09-01 07:00'
    end_time = '2020-09-11 07:00'
    outdoor_cut = outdoor.copy()
    outdoor_cut = outdoor.loc[start_time:end_time]
    
    if name == 'Audubon': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.4207)/1.0739)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242     
    
    else:
            pass
        
    if name == 'Adams': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+0.93)/1.1554)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242   # SRCAA adjustment
    
    else:
            pass

    if name == 'Balboa': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.2878)/1.2457)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242  # SRCAA adjustment
    
    
    else:
            pass      
        
    if name == 'Browne': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.4771)/1.1082)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242    # SRCAA adjustment
    
    
    else:
            pass 

    if name == 'Grant': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+1.0965)/1.29)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242  # SRCAA adjustment
    
    else:
            pass

    if name == 'Jefferson': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+0.7099)/1.1458)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242
    
    else:
            pass

    if name == 'Lidgerwood': 
        outdoor_cut['PM2_5_corrected'] = (outdoor_cut.PM2_5-1.1306)/0.9566*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242 # SRCAA adjustment
    
    else:
            pass

    if name == 'Regal': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5-0.247)/0.9915)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242          # SRCAA adjustment
    
    else:
            pass

    if name == 'Sheridan': 
        outdoor_cut['PM2_5_corrected'] =  ((outdoor_cut.PM2_5+0.6958)/1.1468)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242   # SRCAA adjustment
    
    else:
            pass

    if name == 'Stevens': 
        outdoor_cut['PM2_5_corrected'] = ((outdoor_cut.PM2_5+0.8901)/1.2767)*0.454-outdoor_cut.Rel_humid*0.0483-outdoor_cut.temp*0.0774+4.8242   # SRCAA adjustment
    
    else:
            pass



    return outdoor_cut



























