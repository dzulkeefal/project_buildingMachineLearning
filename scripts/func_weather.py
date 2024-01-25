# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 11:40:37 2024

@author: RDT Simulations
"""

#%%  REMOVE DATA FROM PREVIOUS RUN
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().run_line_magic('reset -f')
except:
    pass

#%%  IMPORT DEPENDENCIES
import pvlib
import numpy as np
from def_classes import case
import sys
#%%  FUNCTIONS

def readWindData(case_weather):
    '''
    Reads the wind speed and wind dir for given time and date
    '''
    month =  case_weather.month
    day = case_weather.day
    hour = case_weather.hour
    
    filename = case_weather.file_weather
    data = pvlib.iotools.read_epw(filename, coerce_year=None)
    
    df = data[0]
    
    hour_data = df.loc[(df['month'] == month) & (df['day'] == day) & (df['hour'] == hour)]
    
    wind_speed = hour_data['wind_speed'].iloc[0]
    wind_direction = hour_data['wind_direction'].iloc[0]
    
    # data2 = np.genfromtxt(filename, skip_header=8, delimiter=",")
    if wind_speed == 0:
        print(f'Wind speed = 0 for {case_weather.day}/{case_weather.month} at {case_weather.hour}:00')
    return wind_speed,wind_direction
    
def updateCutoffMapping_wind(case_weather,cutoffs,field):
    '''
    Update the Cutoff mapping as per the new field and cutoffs list
    '''
    if case_weather.initial_step == True:
        case_weather.cutoffMapping = np.zeros([len(field),len(cutoffs)])
        case_weather.initial_step = False
        
    for icutoff in range(len(cutoffs)):
        for inode in range(len(field)):
            if field[inode] > cutoffs[icutoff]:
                case_weather.cutoffMapping[inode][icutoff] +=1
    return

def calcHoursVoilated_wind(case_weather):
    '''
    Calculate the % of hours that voilate the cutoffs
    '''
    cutoffVoilatedMapping = case_weather.cutoffMapping.copy()
    cutoffVoilatedMapping.fill(0)
    
    cutoffVoilatedMapping = 100*case_weather.cutoffMapping/case_weather.ncases
    
    return cutoffVoilatedMapping
                
#%% MAIN
if __name__ == "__main__":
    inputs = case()
    inputs.month = 8
    inputs.day = 2
    inputs.hour = 17
    inputs.file_weather = fr'C:\ladybug\ESP_Madrid.082210_IWEC\ESP_Madrid.082210_IWEC.epw'
    speed, direction = readWindData(inputs)