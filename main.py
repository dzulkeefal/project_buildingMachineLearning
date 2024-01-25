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
# adding scripts folder to python path
cwd = os.getcwd()
sys.path.append(fr'{cwd}\scripts')

from def_classes import case
from func_interpolationModel import prepareData,trainModel,evaluateModel
from func_vtk import writeFieldVTK
from func_weather import readWindData

#%%  FUNCTIONS 
    
#%% MAIN
# User inputs
inputs = case()
inputs.file_weather = r'C:\ladybug\ESP_Madrid.082210_IWEC\ESP_Madrid.082210_IWEC.epw'
inputs.month = 8                                              
inputs.day  = 23
inputs.hour = 18

inputs.root_data = fr"{cwd}"  
inputs.file_test = ["{}_surfaces_{}".format(inputs.field, str(inputs.angle_new))] 

# Calculations
wind_speed, wind_dir = readWindData (inputs)
# data = prepareData(inputs)
# trainModel(inputs, data)
field = evaluateModel(inputs, wind_speed, wind_dir)
writeFieldVTK(inputs,field,wind_dir)

