#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:08:25 2020

@author: matthew
"""

# make sure to only run this once, if run twice will shift again

def opt_shift(indoor, corr_df):
    
    
    if indoor['Location'].str.contains('Audubon').any():
        shift = corr_df.loc[corr_df['Audubon'].idxmax(), 'offset']                    #       -1
        print('Audubon: Project School; lag = ' , corr_df.loc[corr_df['Audubon'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Audubon'].idxmax(), 'Audubon'])    
    
    elif indoor['Location'].str.contains('Grant').any():
        shift = corr_df.loc[corr_df['Grant'].idxmax(), 'offset']                      #       -3
        print('Grant: Project School; lag = ' , corr_df.loc[corr_df['Grant'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Grant'].idxmax(), 'Grant'])

    elif indoor['Location'].str.contains('Regal').any():
        shift = corr_df.loc[corr_df['Regal'].idxmax(), 'offset']                      #      -5
        print('Regal: Project School; lag = ' , corr_df.loc[corr_df['Regal'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Regal'].idxmax(), 'Regal'])

    elif indoor['Location'].str.contains('Sheridan').any():
        shift = corr_df.loc[corr_df['Sheridan'].idxmax(), 'offset']                   #      -3
        print('Sheridan: Project School; lag = ' , corr_df.loc[corr_df['Sheridan'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Sheridan'].idxmax(), 'Sheridan'])

    elif indoor['Location'].str.contains('Stevens').any():
        shift = corr_df.loc[corr_df['Stevens'].idxmax(), 'offset']                    #      -2
        print('Stevens: Project School; lag = ' , corr_df.loc[corr_df['Stevens'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Stevens'].idxmax(), 'Stevens'])
    
    elif indoor['Location'].str.contains('Adams').any():
        shift = corr_df.loc[corr_df['Adams'].idxmax(), 'offset']                      #       -1
        print('Adams: Non-Project; lag = ' , corr_df.loc[corr_df['Adams'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Adams'].idxmax(), 'Adams'])
    
    elif indoor['Location'].str.contains('Balboa').any():
        shift = corr_df.loc[corr_df['Balboa'].idxmax(), 'offset']                     #       -2
        print('Balboa: Non-Project; lag = ' , corr_df.loc[corr_df['Balboa'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Balboa'].idxmax(), 'Balboa'])
        
    elif indoor['Location'].str.contains('Browne').any():
        shift = corr_df.loc[corr_df['Browne'].idxmax(), 'offset']                     #       -4
        print('Browne: Non-Project; lag = ' , corr_df.loc[corr_df['Browne'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Browne'].idxmax(), 'Browne'])
        
    elif indoor['Location'].str.contains('Jefferson').any():
        shift = corr_df.loc[corr_df['Jefferson'].idxmax(), 'offset']                  #       -3
        print('Jefferson: Non-Project; lag = ' , corr_df.loc[corr_df['Jefferson'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Jefferson'].idxmax(), 'Jefferson'])
        
    elif indoor['Location'].str.contains('Lidgerwood').any():
        shift = corr_df.loc[corr_df['Lidgerwood'].idxmax(), 'offset']                 #       -2
        print('Lidgerwood: Non-Project; lag = ' , corr_df.loc[corr_df['Lidgerwood'].idxmax(), 'offset'],
              'R-value = ', '%.2f' % corr_df.loc[corr_df['Lidgerwood'].idxmax(), 'Lidgerwood'])
    

    indoor['PM2_5_corrected_shift'] = indoor['PM2_5_corrected'].shift(shift)
