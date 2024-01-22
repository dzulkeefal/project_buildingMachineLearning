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
#%%  FUNCTIONS

def readWindData(case_weather):
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
        exit('Wind speed = 0')
    return wind_speed,wind_direction

#%% MAIN
if __name__ == "__main__":
    inputs = case()
    inputs.month = 8
    inputs.day = 2
    inputs.hour = 17
    inputs.file_weather = fr'C:\ladybug\ESP_Madrid.082210_IWEC\ESP_Madrid.082210_IWEC.epw'
    speed, direction = readWindData(inputs)