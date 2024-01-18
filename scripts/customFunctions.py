#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains all the custom functions which are used commonly in other scripts
Types of functions include Array handling, Reading files, Plotting, ROM subroutines, 
Euclidean plane interpolation, Manifold interpolation,

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
# Array Handling
##########################################################################################################################################################
# Normalize an array
def Normalize(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm
    return matrix, norm

# Check if the case is of interpolation, parameter to parameter i.e. in a univariate sense
def IsInterpolation(parameters,test):
    nparams,nv = parameters.shape
    status = [False] * nparams
    for npi in range(nparams):
        minimum = min(parameters[npi,:])
        maximum = max(parameters[npi,:])
        if (test[npi] <= maximum and test[npi] >= minimum): 
            status[npi] = True
    return status

# Check orthogonality of a matrix
def CheckOrthogonality(matrix):
    epsilon = 1e-10
    dotprod = matrix.T.dot(matrix)
    status = abs(dotprod - 1) <= epsilon
    return dotprod, status

# Scale data toi 0-1 range
def ScaleMinMax(data):
    scaledData = (data - np.min(data)) / (np.max(data) - np.min(data))
    return scaledData

# Determines the norm of difference of different datasets from a reference dataset
def L2DiffFromRef(array1, array2, scale):
    # array 1 is the reference array
    nr, nc = array2.shape 
    euclideandist = []
    for i in range(nc):
        dist = LA.norm(array1-array2[:,i])
        euclideandist.append(dist)
    if scale == 1:
        euclideandist = ScaleMinMax(euclideandist)
    min_value = min(euclideandist)
    pref = np.argmin(euclideandist)
    return euclideandist, min_value, pref

# Convert 3 arrays into 1D array as per Femuss requirements
def Convertto1D(array1,array2,array3):
    aux = np.stack((array1,array2,array3),axis=1)
    npoin, ndofn = aux.shape
    array1D = np.reshape(aux[:,:],npoin*3)
    return array1D

# Returns the quadratic intepolation functions for upper and lower surface
def GetAirfoilInterpFunction(airfoilCoord, order='quadratic'):
    rows,columns = airfoilCoord.shape
    # npoints = math.ceil(rows/2)
    splitpoint_aux = np.where(airfoilCoord[:,0] == 0)                              # split upper and lower surface using xcoord=0 location
    splitpoint = splitpoint_aux[0][0]+1
    x_upper = airfoilCoord[0:splitpoint,0]
    y_upper = airfoilCoord[0:splitpoint,1]
    x_lower = airfoilCoord[splitpoint-1:rows,0]
    y_lower = airfoilCoord[splitpoint-1:rows,1]
    f_upper = interpolate.interp1d(x_upper, y_upper, kind= order)
    f_lower = interpolate.interp1d(x_lower, y_lower, kind= order)
    return f_upper, f_lower

# Returns the intepolated y coordinate of airfoil for upper and lower surface
def GetAirfoilInterpCoord(airfoilCoord, xcoord, order='quadratic'):
    # ycoord = [upper surface lower surface]
    ycoord = np.zeros(len(xcoord)*2)
    f_upper, f_lower = GetAirfoilInterpFunction(airfoilCoord,order = order)
    ycoord[0:math.ceil(len(ycoord)/2)] = f_upper([*xcoord])
    ycoord[math.ceil(len(ycoord)/2):] = f_lower([*xcoord])
    return ycoord

def ConvertVectoMat(vec):
    if vec.ndim == 1:
        vec = vec.reshape((vec.size,1))
    return vec

def ConvertUniformtoCosineSpacing (xcoord):
    x_rad = np.linspace(0, math.pi, len(xcoord))
    x_cos = (np.cos(x_rad) / 2) + 0.5
    xcoord= np.sort(x_cos)
    return xcoord

# Convert pressure values to coefficient of pressure cp= 2*(p-p0)/(den*vel^2)
def CalculateCp (pressure, den, vel):
    pressure  = np.array(pressure)
    cp = np.subtract(pressure,0)
    cp = np.multiply(cp, 1/(0.5*den*vel))
    mid = int(len(cp[0,:])/2)
    aux = np.flip(cp[:,mid:],axis= 1)
    cp[:,mid:] = aux
    return cp

def CalculateForceCoefs (forces, den, vel, area): 
    lift_coeff= 2*float(forces[2])  /(den*vel**2*area)
    drag_coeff= 2*float(forces[1]) /(den*vel**2*area)
    return lift_coeff,drag_coeff

# find the peaks and valleys of an oscillating singnal
def FindPeakPoints (array):
    peak_points = []
    for i in range(len(array)-2):
            if (array[i+1] < array[i] and array[i+1] < array[i+2]) or (array[i+1] > array[i] and array[i+1] > array[i+2]):
                peak_points.append(i+1)
    return peak_points

def FindNoOfPeakPointsPerWave (array,peak_points,tol = 0.005):
    if peak_points:
        # find peak points from highest values
        peak_values = array[tuple([peak_points])].T
        max_value = np.max(peak_values)
        max_points = np.where((max_value - peak_values[:])/max_value < tol)[0]
        pts_mean_max = max_points[-1]-max_points[-2]
        
        # find peak points from lowest values. All values are positive
        min_value = np.min(peak_values)
        min_points = np.where((peak_values[:] - min_value)/min_value < tol)[0]
        pts_mean_min = min_points[-1]-min_points[-2]
        
        pts_mean = max(pts_mean_max,pts_mean_min)
    else:
        pts_mean = 0
    return int(pts_mean)

# find the mean value averaged over 5 peak points. If present, multiple peak values are determined
def CalculateMean(array,peak_points,pts_forMean=4):
    period_start = 0
    mean = []
    if array.all() == 0:
        mean.append(0)
    else:
        while period_start+pts_forMean < len(peak_points):
            period_end = period_start+pts_forMean
            # print(array[peak_points[period_start]:peak_points[period_end]])
            mean_aux = np.mean(array[peak_points[period_start]:peak_points[period_end]],axis=0)    
            mean.append(mean_aux)
            period_start = period_end
    return mean 

# find the timeperiod value averaged over 5 peak points. If present, multiple peak values are determined
def CalculateTimePeriod(array,peak_points,pts_forTimePeriod=4):
    period_start = 0
    time_period = []
    while period_start+pts_forTimePeriod < len(peak_points):
        period_end = period_start+pts_forTimePeriod
        time_period.append(array[peak_points[period_end]]- array[peak_points[period_start]])         
        period_start = period_end
        # plt.axvline(x=array[peak_points[period_start],0])
    return time_period

# computes fft amp and freq which can be plotted on a log-log scale
def fft(array,timestep=0.05):
    fourierTransform = np.fft.fft(array)/len(array)                                             # normalize amplitude
    fourierTransform = fourierTransform[range(int(len(array)/2))]                               # exclude sampling frequency
    tpCount     = len(array)
    values      = np.arange(int(tpCount/2))
    samplingFrequency = 1/timestep
    timePeriod  = tpCount/samplingFrequency
    frequencies = values/timePeriod
    
    return fourierTransform, frequencies

# computes the correct RIC including the first basis function as well in contrary to Femuss BE
def computeRIC(s):
    auxt = 0
    auxb = np.zeros(len(s))
    ric = np.zeros(len(s))
    
    auxb[0] = s[0]
    for ibasis in range(1,len(s)):
        auxb[ibasis] = auxb[ibasis-1] + s[ibasis]
        
    auxt = auxb[-1]    
    for ibasis in range(0,len(s)):
        ric[ibasis] = auxb[ibasis]/auxt
    
    return ric
##########################################################################################################################################################
# Reading files
##########################################################################################################################################################

# Read a particular basis number and provide it in 1D as per Femuss
def ReadBasis(filename,no, normalized):
    f1 = h5py.File(filename, 'r')
    basis_ux = np.array(f1.get(f'Velocity X/Basis{no:>4}'))
    basis_uy = np.array(f1.get(f'Velocity Y/Basis{no:>4}'))
    basis_p = np.array(f1.get(f'Pressure/Basis{no:>4}'))
    basis1D = Convertto1D(basis_ux,basis_uy,basis_p)
    if normalized == 1:
        basis1D, norm= Normalize(basis1D)
    f1.close()
    return basis1D

# Read a particular basis number for HyperReduction and provide it in 1D as per Femuss
def ReadHypBasis(filename,iHRarray, no, normalized):
    f1 = h5py.File(filename, 'r')
    basis_ux = np.array(f1.get(f'Velocity X{iHRarray}/Basis{no:>4}'))
    basis_uy = np.array(f1.get(f'Velocity Y{iHRarray}/Basis{no:>4}'))
    basis_p = np.array(f1.get(f'Pressure{iHRarray}/Basis{no:>4}'))
    basis1D = Convertto1D(basis_ux,basis_uy,basis_p)
    if normalized == 1:
        basis1D, norm= Normalize(basis1D)
    f1.close()
    return basis1D

# Read a particular PIProj number for HyperReduction and provide it in 1D as per Femuss
def ReadHypPIProj(filename,iHRarray, no, normalized):
    f1 = h5py.File(filename, 'r')
    PIProj_ux = np.array(f1.get(f'Velocity X{iHRarray}/PIProj{no:>4}'))
    PIProj_uy = np.array(f1.get(f'Velocity Y{iHRarray}/PIProj{no:>4}'))
    PIProj_p = np.array(f1.get(f'Pressure{iHRarray}/PIProj{no:>4}'))
    PIProj1D = Convertto1D(PIProj_ux,PIProj_uy,PIProj_p)
    if normalized == 1:
       PIProj1D, norm= Normalize(PIProj1D)
    f1.close()
    return PIProj1D

# Read mean values
def ReadMean(filename, normalized):
    f1 = h5py.File(filename, 'r')
    mean_ux = np.array(f1.get('SnapshotMean/Velocity X'))
    mean_uy = np.array(f1.get('SnapshotMean/Velocity Y'))
    mean_p = np.array(f1.get('SnapshotMean/Pressure'))  
    if normalized == 1:
        mean_ux, norm= Normalize(mean_ux)
        mean_uy, norm= Normalize(mean_uy)
        mean_p, norm= Normalize(mean_p)
    f1.close()
    return mean_ux, mean_uy, mean_p

def ReadEigenValues(filename):
    f1 = h5py.File(filename, 'r')
    sigma = np.array(f1.get(f'ROM/EigenValues'))
    f1.close()
    return sigma

def WriteEigenValues(f1, sigma):
    f1.create_dataset(f'ROM/EigenValues',data=sigma)
    return 

def WriteRestart(f1, restart):
    f1.create_dataset(f'ROM/Data',data=restart)
    return 

# Write mean values,the file is opened and closed externally
def WriteMean(f1, mean):
    f1.create_dataset(f'SnapshotMean/Velocity X',data=mean[::3])
    f1.create_dataset(f'SnapshotMean/Velocity Y',data=mean[1::3])
    f1.create_dataset(f'SnapshotMean/Pressure',data=mean[2::3])
    return 

# Write basis values, the file is opened and closed externally
def WriteBasis(f1, no, basis):
    # f1 = opened hdf5 file
    # no = basis no
    f1.create_dataset(f'Velocity X/Basis{no:>4}',data=basis[::3])
    f1.create_dataset(f'Velocity Y/Basis{no:>4}',data=basis[1::3])
    f1.create_dataset(f'Pressure/Basis{no:>4}',data=basis[2::3])
    return

# Read a particular basis snapshot and provide it in 1D as per Femuss
def ReadSnapshot(filename,no,normalized):
    f1 = h5py.File(filename, 'r')
    snapshot_ux = np.array(f1.get(f'Snapshots   1/Snapshots{no:>4}'))
    snapshot_uy = np.array(f1.get(f'Snapshots   2/Snapshots{no:>4}'))
    snapshot_p = np.array(f1.get(f'Snapshots   3/Snapshots{no:>4}'))
    snapshot1D = Convertto1D(snapshot_ux,snapshot_uy,snapshot_p)
    if normalized == 1:
        snapshot1D, norm= Normalize(snapshot1D)
    f1.close()
    return snapshot1D

# Read parameter values for interpolation
def ReadParameters(filename):
    param_aux=[]
    with open(filename, 'r') as file:
        for count, line in enumerate(file):
            split_line = line.split(',')
            param_current = [float(i) for i in split_line[:-1]]
            param_aux.append(param_current)
    file.close()
    param_aux=np.array(param_aux)
    return param_aux

# Read airfoil geometry and return it as x=1 to 0 to 1 without repetition
def ReadAirfoilGeom(filename, trailingedge = 'closed'):
    try:
        data = np.loadtxt(filename)
    except:
        print('The airfoil files needs to be corrected manually')
    
    # Check if an airfoil is in this sequence: TE -> upper surface -> LE -> lower surface -> TE 
    flagStart = False
    flagEnd   = False
    if abs(data[0,0]) > abs(data[1,0] - 0.1):
        flagStart = True
    if abs(data[-1,0]) > abs(data[-2,0] - 0.1):
        flagEnd   = True
        
    # Correct the sequence is needed    
    if (flagStart and flagEnd):
        pass
    else:
        rows= data.shape[0]
        mid = int((rows-1)/2)
        # xpts = data[:mid,0]  
        data=np.delete(data,0,axis=0)
        aux = data [:mid,:]
        aux=aux[aux[:,0].argsort()[::-1]]
        data[:mid,:] = aux[:mid,:]
    
    # Ensuring that the trailing edge is closed
    if trailingedge == 'closed':
        data[-1,1] = data[0,1]
        
    return data

def ReadTrackedPoints(filename, also_ycoord=False):
    StartReadingTp = 1
    x_coord = []
    y_coord= []
    for tpfile in open(filename):
        lines_of_tpfile=tpfile.split()
        if lines_of_tpfile[0] == 'POINT':
            StartReadingTp = StartReadingTp + 1
        elif lines_of_tpfile[0] == 'END_TRACKING':
            StartReadingTp = StartReadingTp + 1
        elif StartReadingTp % 2  == 0:
            x_coord.append(float(lines_of_tpfile[0]))
            if also_ycoord:
                y_coord.append(float(lines_of_tpfile[1]))
            
    # the list's x-axis should vary without jump at 1
    mid = int(len(x_coord)/2)
    x_coord[mid:] = reversed(x_coord[mid:])
    if also_ycoord:
        y_coord[mid:] = reversed(y_coord[mid:])
    
    if also_ycoord:
        return x_coord, y_coord
    else:
        return x_coord

# Read pressure at nodes from a file
def ReadPresTp(filename, no_tp, timestep = 'last'):
    counter = 0
    pressure_aux = []
    appendAtTheEnd = False
    with open(filename) as f:
        for line in f:
            lines_of_tpfile=line.split()
            counter = counter+1
            if timestep == 'last':
                appendAtTheEnd = True
                continue
            elif timestep == 'all':
                aux = [float(i) for i in lines_of_tpfile[no_tp*2+1:]]
                pressure_aux.append(aux)
            elif counter <= timestep:
                continue
            else:
                aux = [float(i) for i in lines_of_tpfile[no_tp*2+1:]]
                pressure_aux.append(aux)          
            
        if appendAtTheEnd:
            aux = [float(i) for i in lines_of_tpfile[no_tp*2+1:]]
            pressure_aux.append(aux)
            
    pressure = np.array(pressure_aux)
    return pressure

def ReadForces(filename, timestep = 'last'):
    counter = 0
    forces_aux = []
    appendAtTheEnd = False
    with open(filename) as f:
        for line in f:
            lines_of_forces = line.split()
            counter = counter+1
            if counter <= 2:
                continue
            elif timestep == 'last':
                appendAtTheEnd = True
                continue
            elif timestep == 'all':
                aux = [float(i) for i in lines_of_forces]
                forces_aux.append(aux)
            elif counter <= timestep:
                continue
            else:
                aux = [float(i) for i in lines_of_forces]
                forces_aux.append(aux)          
            
        if appendAtTheEnd:
            aux = [float(i) for i in lines_of_forces]
            forces_aux.append(aux)
    
    forces = np.array(forces_aux)
    return forces

def ReadROMcoord(filename, timestep = 'last'):
    with open(filename) as f:
        for line in f:
            if timestep == 'last':
                pass
        lines_of_rom_coord=line.split()
        rom_coord=[float(i) for i in lines_of_rom_coord[2:]]
    return rom_coord

# Read system residuals from file 
def ReadSystemResiduals(filename, timestep = 'last'):
    sysresid = ReadROMcoord(filename, timestep= timestep)
    return sysresid

# Read the subscales predicted by ANN while testing
def ReadSubscalesPred(filename, inputs, timestep):
    subscales_pred = []
    iread = 0
    with open(filename) as f:
        for line in f:
            lines_of_subscalespred=line.split()
            subscales_aux = [float(i) for i in lines_of_subscalespred[2+inputs:]]
            subscales_pred. append(subscales_aux)
    if timestep == 'last': 
        return subscales_pred[-1]
    else:
        return subscales_pred

# Read last convergence value (2nd last line) for steady case (assuming that velocity is stationary line is not present)
def ReadConvergenceSteady(filename, timestep = 'last'):
    read = 1
    with open(filename) as f:
        for line in (f.readlines() [-2:]):
            if read == 1:
                lines_of_cvg=line.split()
                cvg = [float(i) for i in lines_of_cvg[3:]]
                read = 0
    return cvg

# Save ANN training data to a file
def WriteTrainingDataSteady(filename,params,training_data):
    np.savetxt(filename,np.concatenate((params,training_data), axis = 1))
    return   

##########################################################################################################################################################
# File Hanlding
##########################################################################################################################################################     
    
def noBasisDefect(casesFolder, test_cases, casename):
    import os

    defective = []
    for i in test_cases:
        filename_basis = f"{casesFolder}/{casename}_{i}/{casename}.gid/rst/{casename}.rom.bas"
        
        exists = os.path.isfile(filename_basis)   
        if exists:
            pass
        else:
            defective.append(i)
    return defective 

def checkConvergence(casesFolder, test_cases, casename, tol=1e-7):
    cvg = [] 
    for i in test_cases:       
        filename_convergence = f"{casesFolder}/{casename}_{i}/{casename}.gid/results/{casename}0.nsi.cvg"
        
        cvg_aux = ReadConvergenceSteady(filename_convergence)
        if cvg_aux[0] > tol or cvg_aux[0] == 0:
            print (f'{i} not converged')
        cvg.append(cvg_aux)
    return cvg


def deleteFilesMultCases(filepath, cases, casename, casesFolder):
    import os
    
    for i in range(len(cases)):
        command = f'rm {casesFolder}/{casename}_{cases[i]}/{casename}.gid/{filepath}'
        os.system(command)
    return

def copyFileToMultCases(filepath, cases, casename, casesFolder):
    '''
    compies a file to all the cases. Based on the name and extension, the destination folder is decided.

    Parameters
    ----------
    filepath : TYPE
        file name with its path.
    cases : TYPE
        all the cases with casesID e.g. naca0012.
    casename : TYPE
        generic case name e.g. Airfoil.
    casesFolder : TYPE
        folder which contains all the cases.

    Returns
    -------
    None.

    '''
    
    import os
    
    for i in range(len(cases)):
        path1 = filepath
        if path1 == 'run.sh':
            path2 = f'{casesFolder}/{casename}_{cases[i]}/{casename}.gid/'
        elif path1 == "jobInfo.py":
            path2 = f'{casesFolder}/{casename}_{cases[i]}/{casename}.gid/'
        elif path1 == 'resubmitTheJobIfNeeded.py':
            path2 = f'{casesFolder}/{casename}_{cases[i]}/{casename}.gid/'
        elif path1 == 'multipleTries.sh':
            path2 = f'{casesFolder}/{casename}_{cases[i]}/{casename}.gid/'
        elif path1 == f'{casename}.nsi.rst':
            path2 = f'{casesFolder}/{casename}_{cases[i]}/{casename}.gid/rst/'
        else:
            path2 = f'{casesFolder}/{casename}_{cases[i]}/{casename}.gid/data/'

        command = f'cp {path1} {path2}'
        os.system(command)
    return
        
def copyFilesFromMultCases(filepath, cases, casename, casesFolder):
    '''
    copies a particular file from all cases to a specific location

    Parameters
    ----------
    filepath : TYPE
        location of file within .gid folder.
    cases : TYPE
        all the cases with casesID e.g. naca0012.
    casename : TYPE
        generic case name e.g. Airfoil.
    casesFolder : TYPE
        folder which contains all the cases.

    Returns
    -------
    None.

    '''
    import os
    
    for i in range(len(cases)):
        path1 = f'{casesFolder}/{casename}_{cases[i]}/{casename}.gid/{filepath}'
        path2 = f'{casesFolder}/plots/forces_{cases[i]}'
        
        command = f'cp {path1} {path2}'
        os.system(command)
    return

def UpdateJobName(casesFolder,casename,caseID): 
    import os
    # caseFolder  : the folder of the cases
    # casename    : the common name of the multiple case scenario e.g. Airfoil
    # caseID         : identifier for this particular case
    
    os.chdir(f'{casesFolder}/{casename}_{caseID}/{casename}.gid/')
    
    
    newLUMIrunfile=''
    for LUMIrunfile in open(f"run.sh"):
        lines_of_LUMIrunfile=LUMIrunfile.split()
        aux=[]
        if len(lines_of_LUMIrunfile) >1 and lines_of_LUMIrunfile[1] == f'--job-name={casename}':
            aux.append(LUMIrunfile.replace(f'--job-name={casename}',f'--job-name={casename}_{caseID}'))
        else:
            aux.append(LUMIrunfile)
        
        for c in aux:
            newLUMIrunfile+=str(c)
    
    OverWritingLUMIrunfile = open(f"run.sh",'w')
    OverWritingLUMIrunfile.write(newLUMIrunfile)
    OverWritingLUMIrunfile.close()

##########################################################################################################################################################
# Find Defective Cases
##########################################################################################################################################################    

def ALEFailedDefect(casesFolder, test_cases, casename):
    import os
    
    defective = []
    for i in test_cases:
        filename_glog = f"{casesFolder}/{casename}_{i}/{casename}.gid/results/{casename}0.glog"
        
        isNotEmpty = os.path.getsize(filename_glog) > 0   
        if isNotEmpty:
            pass
        else:
            defective.append(i)
    return defective 

# Check if the case did not run on Lumi due to interconnectFailed Error
def interrconnectFailedDefect(casesFolder, test_cases, casename):
    import os
    import glob
    
    defective = []
    for i in test_cases:
        folder = f"{casesFolder}/{casename}_{i}/{casename}.gid/"
        os.chdir(folder)
        filename_LUMIerr = glob.glob("*.err")
        read = 1
        for errFile in open(filename_LUMIerr[0]):
            lines_of_errFile=errFile.split()
            if read == 1 and len(lines_of_errFile) > 2 and lines_of_errFile[-1] == 'interconnect':
                defective.append(i)
                read = 0
        os.chdir(casesFolder)
    return defective 

# Check if the result folder is empty impling that the case did not run, probably due to interconnectFailed Error
def resultsFolderEmptyDefect(casesFolder, test_cases, casename):
    import os
    
    defective = []
    for i in test_cases:
        folder = f"{casesFolder}/{casename}_{i}/{casename}.gid/results"
        if not os.listdir(folder):
                defective.append(i)
    return defective 

# Read last timestep value (3rd last line) incase velocity is stationary line is present
def checkNaNDefect(casesFolder, test_cases, casename):
    import os
    import math
    
    defective = []
    cvg = []
    for i in test_cases:
        filename_cvg = f"{casesFolder}/{casename}_{i}/{casename}.gid/results/{casename}0.nsi.cvg"    
        read = 1
        with open(filename_cvg) as f:
            for line in (f.readlines() [-3:]):
                if read == 1:
                    lines_of_cvg=line.split()
                    cvg_aux = float(lines_of_cvg[3])
                    cvg.append(cvg_aux)
                    
                    if math.isnan(cvg_aux):
                        defective.append(i)
                        
                    read = 0
        

    return defective, cvg

# Check if the case timedOut
def timedOutDefect(casesFolder, test_cases, casename):
    import os
    import glob
    
    defective = []
    for i in test_cases:
        folder = f"{casesFolder}/{casename}_{i}/{casename}.gid/"
        os.chdir(folder)
        filename_LUMIerr = glob.glob("*.err")
        read = 1
        for errFile in open(filename_LUMIerr[0]):
            lines_of_errFile=errFile.split()
            if read == 1 and len(lines_of_errFile) > 2 and lines_of_errFile[-2] == 'LIMIT':
                defective.append(i)
                read = 0
        os.chdir(casesFolder)
    return defective    
    
##########################################################################################################################################################
# Plotting
##########################################################################################################################################################

def PlotParameters(parameters):
    nparams,nv = parameters.shape
    
    # Plotting value of a given param vs airfoil no. Different curves represent different params
    x = np.linspace(1,nv+1,nv)
    plt.figure()
    for npi in range(nparams):
        plt.plot(x,parameters[npi,:],label = npi)
    plt.legend(loc='right',bbox_to_anchor=(1.35, 0.5),ncol=1)
    plt.savefig('parameters_variation', dpi=400, bbox_inches='tight')     
    
    # Plotting param value vs param number. Different curves represent different airfoils
    x = np.linspace(1,nparams+1,nparams)
    plt.figure()
    for nvi in range(nv):
        plt.plot(x,parameters[:,nvi])
    plt.savefig('parametesOfAirfoils', dpi=400, bbox_inches='tight')
    return

##########################################################################################################################################################
# Reduced Order Modeling
##########################################################################################################################################################

# Projection on the reduced space to find reduced coordinates
def ProjOnROM (Y, basis):
    # Y = Y_FOM - Y_mean
    reduced_coef = np.dot(basis.T,Y) 
    return reduced_coef

# Injection to the full order space
def InjectOnFOM(reduced_coef, basis):
    Y_FOM = np.dot(basis,reduced_coef) 
    return Y_FOM

##########################################################################################################################################################
# Interpolation schemes on the Tanget/Euclidean plane
##########################################################################################################################################################

# RBF for tangent space interpolation
def RbfInt(pTrain,valuesTrain,pTest):
    nx,nc = valuesTrain.shape
    valuesTest = np.zeros(nx)
    x = ([pTrain[j,:]for j in range(len(pTrain[:,0]))])
    for nxi in range(nx):
        rbfi = Rbf(*x, valuesTrain[nxi,:])   
        valuesTest[nxi] = rbfi(*pTest) 
    return valuesTest

# Piecewise linear for tangent space interpolation
def PwInt(pTrain,valuesTrain,pTest):
    nx,nc = valuesTrain.shape
    valuesTest = np.zeros(nx)
    x = ([pTrain[:,j]for j in range(len(pTrain[0,:]))])
    for nxi in range(nx):
        interp = LinearNDInterpolator(x, valuesTrain[nxi,:],fill_value = 1)   
        valuesTest[nxi] = interp(*pTest) 
    return valuesTest

# Nearest-neighbor interpolation for tangent space interpolation
def NearestndInt(pTrain,valuesTrain,pTest):
    nx,nc = valuesTrain.shape
    valuesTest = np.zeros(nx)
    x = ([pTrain[:,j]for j in range(len(pTrain[0,:]))])
    for nxi in range(nx):
        interp = NearestNDInterpolator(x, valuesTrain[nxi,:])   
        valuesTest[nxi] = interp(*pTest) 
    return valuesTest

def LagInt(pTrain,valuesTrain,pTest):
    nx,nc = valuesTrain.shape
    alpha = np.zerps(nc)
    valuesTest = np.zeros(nx)   
    for i in range(nc):
        for j in range(nc):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-pTrain[j])/(pTrain[i]-pTrain[j])
    for i in range(nc-1):
        valuesTest = valuesTest + alpha[i] * valuesTrain[:,i]
    return valuesTest

def TangentInt(p,Gamma,pTest,EuclideanInt):
    if EuclideanInt == 'rbf':   
        interpolatedValues = RbfInt(p,Gamma,pTest)
    elif EuclideanInt == 'pw':    
        interpolatedValues = PwInt(p,Gamma,pTest)
    elif EuclideanInt == 'nearest':    
        interpolatedValues = NearestndInt(p,Gamma,pTest)
    elif EuclideanInt == 'lagrange':    
        interpolatedValues = LagInt(p,Gamma,pTest,EuclideanInt)
    return interpolatedValues     

##########################################################################################################################################################
# Interpolation schemes on a manifold
##########################################################################################################################################################

# For Orthogonal matrices: Grassman Manifold
def GrassmannInt(Phi,pref,p,pTest,EuclideanInt):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    # EuclideanInt is the euclidean interpolation to be used. Options include rbf, pw, nearest, lagrange
    
    nx,nc = Phi.shape
    Phi0 = Phi[:,pref] 

    print('calculating Gamma')
    Gamma = np.zeros([nx,nc])
    outerdot = np.outer(Phi0,Phi0)                                   # subtract identity wihtout creating the matrix
    for j in range(nx):
        outerdot[j,j] = 1 - outerdot[j,j]
        
    for i in range(nc):
        auxl = outerdot.dot(Phi[:,i])
        auxr = 1/( Phi0.T.dot(Phi[:,i]) )
        aux = auxl*auxr
               
        # U, S, Vh = LA.svd(temp, full_matrices=False)
        U, S = Normalize(aux) 
        Vh = 1
        Gamma[:,i] = U*math.atan(S)
    
    # Tangent space interpolation
    print('calculating GammaL')
    
    GammaL = TangentInt(p,Gamma,pTest,EuclideanInt)    
            
    interpolation = IsInterpolation(Gamma, GammaL) 

    print('calculating PhiL')        
    # U, S, Vh = LA.svd(GammaL, full_matrices=False)
    U, S = Normalize(GammaL)
    PhiL = Phi0*math.cos(S)+U*math.sin(S)
   
    return PhiL

# For R^{M x N} as per Farhat's paper
def ManifoldIntReal(Phi,pref,p,pTest,EuclideanInt):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    # EuclideanInt is the euclidean interpolation to be used. Options include rbf, pw, nearest, lagrange

    nx,nc = Phi.shape
    Phi0 = Phi[:,pref] 

    print('calculating Gamma')
    Gamma = np.zeros([nx,nc])
       
    for i in range(nc):
        Gamma[:,i] = Phi[:,i] - Phi0
    
    # Tangent space interpolation
    print('calculating GammaL')

    GammaL = TangentInt(p,Gamma,pTest,EuclideanInt)    
    
    interpolation = IsInterpolation(Gamma, GammaL) 

    print('calculating PhiL')        
    PhiL = Phi0 + GammaL
   
    return PhiL
#############################################################################
