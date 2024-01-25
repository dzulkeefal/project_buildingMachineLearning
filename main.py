# -*- coding: utf-8 -*-

"""
Created on Fri Mar  3 20:22:02 2023

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
import sys
import os
import numpy as np
# adding scripts folder to python path
cwd = os.path.dirname(__file__)
sys.path.append(fr'{cwd}\scripts')

from def_classes import case
from func_interpolationModel import prepareData,trainModel,evaluateModel,readTrainedData
from func_vtk import writeField_vtk,writeCutOffMap_vtk
from func_weather import readWindData,updateCutoffMapping_wind,calcHoursVoilated_wind

#%%  FUNCTIONS 
    
#%% MAIN
# User inputs
file_weather = r'C:\ladybug\ESP_Madrid.082210_IWEC\ESP_Madrid.082210_IWEC.epw'
months = np.arange(1,13,1)                                              
days  = np.arange(1,29,1)
hours = np.arange(1,24,3)
wind_cutoffs = [5,10,15]                                               # list of wind cutoff in m/s

# Calculations
ncases = len(months)*len(days)*len(hours)
ncutoffs = len(wind_cutoffs)
root_data = fr"{cwd}"  
current_case = case(file_weather,root_data, ncases,ncutoffs)
current_case.file_test = ["{}_surfaces_{}".format(current_case.field, str(current_case.angle_new))] 

for current_case.month in months:
    for current_case.day in days:
        for current_case.hour in hours:
            print (f'Calculating for {current_case.day}/{current_case.month} at {current_case.hour}:00')
            wind_speed, wind_dir = readWindData (current_case)
            # data = prepareData(current_case)
            # trainModel(current_case, data)
            readTrainedData(current_case)
            field = evaluateModel(current_case, wind_speed, wind_dir)
            updateCutoffMapping_wind(current_case,wind_cutoffs,field) 
            
cutOffVoilationMap = calcHoursVoilated_wind(current_case)
# writeField_vtk(current_case,field,wind_dir)
writeCutOffMap_vtk(current_case,cutOffVoilationMap,wind_cutoffs)

