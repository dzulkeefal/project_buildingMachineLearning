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
    # Variables
    field = "U"                                            # fields available: U, P
    angle_new = 0                                          # the current angle being analysed
    wind_ref = 5                                           # reference wind speed in m/s 
    ncases = 0                                             # total nummber of cases to be analysed
    initial_step = True                                    # checking for the first run to initialize some var if needed
    icase = 0                                              # case number of the current case in the loop
    
    grados = [degree_div_10*10 for degree_div_10 in list(range(37))]                         # angles available
    interp_method = 'Rbf'                                  # interp_quad, lagrange,Rbf
    f = []                                                 # list of interpolation func for ROM coefficients
    
    use_SVD = True                                         # perform SVD or not
    subtract_mean = True                                   # subtract mean for SVD
    ric = 1.00                                             # information content to retain in the POD modes 
    dofr = 0                                               # ROM dofs
    s = None                                               # SVD eigen values
    u = None                                               # SVD basis
    A_coef = None                                          # ROM coefs for training cases
    data_training_mean = None                              # mean value of snapshots
    
    
    mesh_interpolation = 'linear'                          # interpolation to be used by griddata
    fillValue = 1000                                       # random high value to fill-in my mesh interpolator
    npoins = 500                                           # no of points for created Structured reference mesh
    rotateMesh = False                                     # rotate mesh to have similar inlet veloc direction
    useRefMeshForProjection = False                        # project case meshes to a common ref mesh
    coords_ref = None                                      # coords for reference mesh
    boundaryNodes = None                                   # boundary nodes (no Slip) for a case
    
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
    cutoffMapping = None                                   # contains the cutoff mapping for all nodes and cutoffs
    
    #grashopper
    gh_grid = None                                         # nodes of grass hopper mesh
    file_ghGrid = 'ghGridCoords.csv'                       # file containg nodes of gh mesh
    file_windGh = 'ghWindFromCFD.csv'                      # file containing field from CFD for gh. Each col corresponds to a gh time step
    gh_field = None                                        # field from CFD to be used by gh
    
    # Functions
    
    def __init__(self,file_weather=None,root_data=None, ncases=None,ncutoffs=None):
        self.file_weather = file_weather
        self.root_data = root_data
        self.ncases = ncases
        self.ncutoffs = ncutoffs
                