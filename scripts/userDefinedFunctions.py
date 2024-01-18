#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains all user defined functions

@author: dzulkeefal
"""
#%% IMPORT DEPENDENCIES 
import numpy as np
from numpy import linalg as LA
import h5py
from scipy import interpolate
import matplotlib.pyplot as plt
import math
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import os

#%% FUNCTION DEFINITIONS

##########################################################################################################################################################
# POD for Urban Flow Fields
##########################################################################################################################################################

