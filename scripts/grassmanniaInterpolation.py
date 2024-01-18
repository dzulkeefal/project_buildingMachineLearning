# -*- coding: utf-8 -*-
"""
This script computes the Grassmanian interpolation
"""

# remove all variable from the previous run if on spyder 
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#%%--- IMPORT DEPENDENCIES ------------------------------------------------------+
import numpy as np
from numpy import linalg as LA
import h5py
from scipy.interpolate import Rbf
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
import math
from scipy.stats.stats import pearsonr   
from customFunctions import * # Normalize, IsInterpolation, CheckOrthogonality, ScaleScaleMinMax, L2DiffFromRef, Convertto1D
import os

#%% Define Functions

# Interpolation schemes on the tanget plane
#############################################################################

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
    x = ([pTrain[j,:]for j in range(len(pTrain[:,0]))])
    
    for i in range(nc):
        for j in range(nc):
            if (j != i) :
                alpha[i] = alpha[i]*(pTest-p[j])/(p[i]-p[j])
    for i in range(nc-1):
        GammaL = GammaL + alpha[i] * Gamma[:,:,i]
        
    for nxi in range(nx):
        rbfi = Rbf(*x, valuesTrain[nxi,:])   
        valuesTest[nxi] = rbfi(*pTest) 
    return valuesTest
#############################################################################

# Grassmann Manifold Interpolation
#############################################################################ç
def GrassmannInt(Phi,pref,p,pTest):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    
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
    GammaL = RbfInt(p,Gamma,pTest)
    # GammaL = PwInt(p,Gamma,pTest)
    # GammaL = NearestndInt(p,Gamma,pTest)
    
    interpolation = IsInterpolation(Gamma, GammaL) 

    print('calculating PhiL')        
    # U, S, Vh = LA.svd(GammaL, full_matrices=False)
    U, S = Normalize(GammaL)
    PhiL = Phi0*math.cos(S)+U*math.sin(S)
   
    return PhiL

def ManifoldInt(Phi,pref,p,pTest):
    # Phi is input basis [training]
    # pref is the reference basis [arbitrarty] for Grassmann interpolation
    # p is the set of training parameters
    # pTest is the testing parameter
    
    nx,nc = Phi.shape
    Phi0 = Phi[:,pref] 

    print('calculating Gamma')
    Gamma = np.zeros([nx,nc])
       
    for i in range(nc):
        Gamma[:,i] = Phi[:,i] - Phi0
    
    # Tangent space interpolation
    print('calculating GammaL')
    GammaL = RbfInt(p,Gamma,pTest)
    # GammaL = PwInt(p,Gamma,pTest)
    # GammaL = NearestndInt(p,Gamma,pTest)
    
    interpolation = IsInterpolation(Gamma, GammaL) 

    print('calculating PhiL')        
    PhiL = Phi0 + GammaL
   
    return PhiL
#############################################################################

#%% MAIN

m_all=["n0012", "n63412", "raf48", "n63215b","n0010","n0015","n1410","n2414","n4412","n4415","rae104","rae5212","rae5215","SC(2)-0412","SC(2)-0612","rae2822"]
j = [4,5,0]
# j=[x for x in range(len(m_all))]
m = [m_all[k] for k in j]
total_cases = len(m)
bezier_points = 3
ptest = len(m)-1                                                                      #last airfoil
npoin = 9661
r = 6                                                                                #pres,velx,vely,pres_mean,velx_mean,vely_mean                                                                 
basis_file = "rae2822_new"

# Initializing
basis = np.zeros([npoin*3,total_cases]) 
ux_mean = np.zeros([npoin,total_cases])  
uy_mean = np.zeros([npoin,total_cases])  
p_mean = np.zeros([npoin,total_cases])  
mean1D = np.zeros([npoin*3,total_cases]) 
snapshots = np.zeros([npoin*3,total_cases])
params = np.zeros([bezier_points*4+1,total_cases])

cwd = os.path.dirname(os.path.abspath(__file__))                                        #determine current working directory
  
# Reading data
xcoord = np.arange(0.02,0.99,0.01)
airfoil_ycoords = np.zeros([len(xcoord)*2,total_cases]) 
for i in range(total_cases):
    ux_mean[:,i], uy_mean[:,i], p_mean[:,i] = ReadMean(f'{cwd}/basis/{m[i]}',normalized=0)
    basis[:,i] = ReadBasis(f'{cwd}/basis/{m[i]}',no=1,normalized=0)                             
    snapshots[:,i] = ReadSnapshot(f'{cwd}/snapshots/{m[i]}',no=100,normalized=0)
    params[:,i] = ReadParameters(f'{cwd}/controlPoints/controlPoints_{m[i]}.txt')
    mean1D[:,i] = Convertto1D(ux_mean[:,i], uy_mean[:,i], p_mean[:,i])
    
    coords = ReadAirfoilGeom(f'{cwd}/airfoilGeom/{m[i]}.txt')
    airfoil_ycoords[:,i] = GetAirfoilInterpCoord(coords, xcoord)

plt.figure()
# plotting parameters
for i in range(total_cases):
    plt.plot(params[1::2,i],params[2::2,i],label=m[i])
plt.legend()    

# Check Orthogonality
dot_prod,orthogonality = CheckOrthogonality(basis[:,0])

# # Plotting parameters
# PlotParameters(params)

# Checking if it is an interpolation case
isInterp = IsInterpolation(params[:,:-1],params[:,ptest])

# Find the closest pref using Euclidean dist of params
euclideandist_param, minval_param, pref_param = L2DiffFromRef(params[:,ptest],params[:,:-1],scale=1)
euclideandist_snap, minval_snap, pref_snap = L2DiffFromRef(snapshots[:,ptest],snapshots[:,:-1],scale=0)
euclideandist_basis, minval_basis, pref_basis = L2DiffFromRef(basis[:,ptest],basis[:,:-1],scale=1)
euclideandist_mean, minval_mean, pref_mean = L2DiffFromRef(mean1D[:,ptest],mean1D[:,:-1],scale=1)
euclideandist_coord, minval_coord, pref_coord = L2DiffFromRef(airfoil_ycoords[:,ptest],airfoil_ycoords[:,:-1],scale=0)    
# Check correlation between shape and params
corr_coordparam = (pearsonr(euclideandist_coord,euclideandist_param))
corr_coordsnap = (pearsonr(euclideandist_coord,euclideandist_snap))
# Check correlation with snapshots
corr_param = (pearsonr(euclideandist_snap,euclideandist_param))
corr_basis = (pearsonr(euclideandist_snap,euclideandist_basis))
corr_mean = (pearsonr(euclideandist_snap,euclideandist_mean))
x = np.arange(1,len(euclideandist_param)+1)
# Plotting different euclidean distances
# plt.plot (x,euclideandist_param,label='param')
# plt.plot (x,euclideandist_basis,label='basis')
plt.plot (x,euclideandist_snap,label='snap')
# plt.plot (x,euclideandist_mean,label='mean')
plt.plot (x,euclideandist_coord,label='coord')
plt.legend()
plt.savefig('EuclideanDist', dpi=400, bbox_inches='tight')
plt.show


norm_pmean = [np.linalg.norm(p_mean[:,j]) for j in range(total_cases)]
norm_basis = [np.linalg.norm(basis[:,j]) for j in range(total_cases)]

# xyz
# Perform the Grassmanian interpolation and scale  
pref = pref_coord 
# basis_int = GrassmannInt(basis[:,:-1],pref,params[:,:-1],params[:,ptest])
# basis_int,norm = Normalize(basis_int)
# ux_mean_int = GrassmannInt(ux_mean[:,:-1],pref,params[:,:-1],params[:,ptest])
# ux_mean_int = ux_mean_int*(np.linalg.norm(ux_mean[:,pref])/np.linalg.norm(ux_mean_int)) #scaling as per the reference ponit
# uy_mean_int = GrassmannInt(uy_mean[:,:-1],pref,params[:,:-1],params[:,ptest])
# uy_mean_int = uy_mean_int*(np.linalg.norm(uy_mean[:,pref])/np.linalg.norm(uy_mean_int)) #scaling as per the reference ponit
# p_mean_int = GrassmannInt(p_mean[:,:-1],pref,params[:,:-1],params[:,ptest])
# p_mean_int = p_mean_int*(np.linalg.norm(p_mean[:,pref])/np.linalg.norm(p_mean_int))     #scaling as per the reference ponit
# mean1D_int = GrassmannInt(mean1D[:,:-1],pref,params[:,:-1],params[:,ptest])
snapshots_int = ManifoldInt(snapshots[:,:-1],pref,params[:,:-1],params[:,ptest])
# snapshots_int = RbfInt(params[:,:-1],snapshots[:,:-1],params[:,ptest])

# correlation_pmean = pearsonr(p_mean_int,p_mean[:,ptest])
# correlation_uxmean = pearsonr(ux_mean_int,ux_mean[:,ptest])
# correlation_uymean = pearsonr(uy_mean_int,uy_mean[:,ptest])
# correlation_basis = pearsonr(basis_int,basis[:,ptest])
# correlation_mean1D = pearsonr(mean1D_int,mean1D[:,ptest])
correlation_snapshots = pearsonr(snapshots_int[:],snapshots[:,ptest])

# normratio_pmean = np.linalg.norm(p_mean[:,ptest])/np.linalg.norm(p_mean_int)
# normratio_uxmean = np.linalg.norm(ux_mean[:,ptest])/np.linalg.norm(ux_mean_int)
# normratio_uymean = np.linalg.norm(uy_mean[:,ptest])/np.linalg.norm(uy_mean_int)
# normratio_mean1D = np.linalg.norm(mean1D[:,ptest])/np.linalg.norm(mean1D_int)
# mean1D_int = mean1D_int*normratio_mean1D
normratio_snapshots = np.linalg.norm(snapshots[:,ptest])/np.linalg.norm(snapshots_int)
snapshots_int = snapshots_int*normratio_snapshots

rel_error = np.linalg.norm(snapshots[:,ptest]-snapshots_int)
# xyz
# Combining all the data in a matrix of desired shape for ease of writing
# Writing 0s for mean and snap_int for basis_1
datasetaux = np.reshape(snapshots_int, [npoin,3])
datasetmeanaux = np.zeros([npoin,3])
# dataset = np.stack((p_mean_int,ux_mean_int,uy_mean_int),axis=1)
dataset = np.concatenate((datasetmeanaux,datasetaux),axis=1)

# Save the new basis
f1 = h5py.File(basis_file, 'r+')                                                      # open the file
heading = [ 'SnapshotMean/Velocity X', 'SnapshotMean/Velocity Y', 'SnapshotMean/Pressure', 'Velocity X/Basis   1', 'Velocity Y/Basis   1','Pressure/Basis   1',]
for i in range(len(heading)):
    data = f1[heading[i]]                                                             #load the data
    data[...] = dataset[:,i]                                                          #assign new values to data
f1.close()   

# Checking if the new basis are written correctly
f1 = h5py.File(basis_file, 'r')
for i in range(len(heading)):
    new_value = np.array(f1.get(heading[i]))
    status = np.allclose(new_value, dataset[:,i] )
    print(status)
f1.close()