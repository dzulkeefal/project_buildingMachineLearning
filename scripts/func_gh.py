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
import numpy as np
from func_vtk import files_vtk_to_array
from func_interpolationModel import projectFieldToNewMesh

#%%  FUNCTIONS
def readGhGrid(gh_case):
    '''
    reads gh mesh coords in 2D, the mesh is assumed to be on a plane
    '''
    root_data = fr'{gh_case.root_data}\{gh_case.folder_evaluation}'  
    file_mesh = fr'{root_data}\{gh_case.file_ghGrid}'
    gh_grid = np.loadtxt(file_mesh,delimiter= ',')
    gh_case.gh_grid = gh_grid[:,:2].copy()
    return

def writeFieldForGrasshopper(gh_case):
    '''
    saves the CFD results on gh mesh
    '''
    root_data = fr'{gh_case.root_data}\{gh_case.folder_evaluation}'  
    file_results = fr'{root_data}\{gh_case.file_windGh}'
    np.savetxt(file_results, gh_case.gh_field, delimiter = ",",fmt='%10.5f')
    return

def calculateFieldForGrasshopper(gh_case,field):
    '''
    calculate field values on the grid provided by grassHopper
    writes it when last case is reached
    '''
    root_data = fr'{gh_case.root_data}\{gh_case.folder_evaluation}'  
    if gh_case.gh_grid is None:
        readGhGrid(gh_case)
    
    _, coords_old = files_vtk_to_array(root_data, [f'{gh_case.field}_reference_0'],getCoords=True)
    field_new = projectFieldToNewMesh(coords_old, field, gh_case.gh_grid, gh_case.mesh_interpolation, gh_case.fillValue,correct_FillValues= False)
    
    try: 
        if gh_case.gh_field == None:
            gh_case.gh_field = np.zeros([gh_case.gh_grid.shape[0],gh_case.ncases])
    except: 
        pass
        
    gh_case.gh_field[:,gh_case.icase] = field_new[:,0]
    
    if gh_case.icase == gh_case.ncases-1:
        writeFieldForGrasshopper(gh_case)
        
    return   
    

              
#%% MAIN
if __name__ == "__main__":
    pass