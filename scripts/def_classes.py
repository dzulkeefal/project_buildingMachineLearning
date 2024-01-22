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

#%%  FUNCTIONS 
    
#%% CLASS DEFINITIONS
class case:
    # User inputs
    field = "U"                                             # fields available: U, P
    angle_new = 0
    wind_ref = 5                                           # reference wind speed in m/s 
    use_SVD = True                                         # perform SVD or not
    subtract_mean = True                                   # subtract mean for SVD
    interp_method = 'Rbf'                                  # interp_quad, lagrange,Rbf
    grados = [degree_div_10*10 for degree_div_10 in list(range(37))]                         # angles available
    fillValue = 1000                                       # random high value to fill-in my mesh interpolator
    ric = 1.00                                             # information content to retain in the POD modes 
    mesh_interpolation = 'linear'                          # interpolation to be used by griddata
    npoins = 500                                           # no of points for created Structured reference mesh
    rotateMesh = False                                     # rotate mesh to have similar inlet veloc direction
    useRefMeshForProjection = False                        # project case meshes to a common ref mesh
    folder_training = 'training_urbanExample'
    folder_evaluation = 'evaluation_urbanExample'                                  
    root_data = ""                                         # folder containing the training and evaluate folders
    file_test = []                                         # test file as per the angle_new
    file_weather = ''                                      # weather file to read the data
    file_refMesh = 'refMesh.txt'                           # file for saving reference mesh
    file_basis = 'basis'                                   # file for saving basis,mean and eigen values
    file_coefs = 'coef.txt'                                # file for ROM coeffs
    file_boun = 'boun.txt'                                 # file for boundary nodes
    # evaluation date and hour
    month = 0                                              
    day = 0
    hour = 0