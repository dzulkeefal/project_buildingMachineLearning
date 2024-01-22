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
import os, glob, shutil
# adding scripts folder to python path
cwd = os.getcwd()
sys.path.append(fr'{cwd}')

import numpy as np
import func_vtk as f_vtk
# import svd_interp as svd_op
# import plotting_libr as plt_lib
# import vtk
# from vtk.util.numpy_support import vtk_to_numpy
from scipy import interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
from scipy.interpolate import lagrange, Rbf
import h5py

#%%  FUNCTIONS

def findNodesWithFillValue(array,fillValue):
    '''
    Finds the index of nodes which have been inside fillValue i.e. which griddata could not interpolate    
    '''
    counter = 0
    index_list = []
    for value in array:
        if value == fillValue:
            index_list.append(counter)
        counter += 1
    return index_list
    
def plotCoords(coords):
    plt.scatter(coords[:,0],coords[:,1])
    plt.show()
    
def correctFillValues(coords_old, array, coords_new):
    '''
    Corrects fillValues using a better method; assing the nearest node value in this case    
    '''
    array = griddata(coords_old, array, coords_new, method='nearest', rescale=True)   
    return array

def projectFieldToNewMesh(coords_old, array, coords_new, meshInterpolationMethod ,fill_value, rescale=True,correct_FillValues= True):
    '''
    Project to a new mesh and also provides the option to correct fillValues    
    '''
    array_interpolated = griddata(coords_old, array, coords_new, method=meshInterpolationMethod ,fill_value=fill_value, rescale=True)   

    # nodes_withFillValues = findNodesWithFillValue(array_interpolated, fillValue)
    # plotCoords(coords_new[nodes_withFillValues,:])
    
    if correct_FillValues:
        # Exchange the fill values with the nearest values
        fillValueIndex = findNodesWithFillValue(array_interpolated, fill_value)
        # plotCoords(coords_ref[fillValueIndex,:])
        if fillValueIndex:
            array_correctFillValues = correctFillValues(coords_old, array, coords_new[fillValueIndex,:])
            array_interpolated[fillValueIndex] = array_correctFillValues
        
    return array_interpolated

def createStructuredMesh(bounds, npoints, shape):
    '''
    Creates structured circular or rectangular mesh center at (0,0) without duplciate nodes    
    '''
    if shape == 'circular':
        r = np.linspace(0, bounds, npoints)
        theta = np.linspace(0, 2*np.pi, npoints)
        r, theta = np.meshgrid(r, theta)
    
        X = r * np.sin(theta)
        Y = r * np.cos(theta)
    
    elif shape == 'rectangular':
        x = np.linspace(-bounds, bounds, npoints)
        y = np.linspace(-bounds, bounds, npoints)
        X,Y = np.meshgrid(x, y)

    coords = []
    for a, b in  zip(X, Y):
        for a1, b1 in zip(a, b):
            coords.append((a1, b1,))
            
    # Remove the duplicates
    coords_withoutDuplicates = [list(t) for t in set(element for element in coords)]
    coords_withoutDuplicates = np.array(coords_withoutDuplicates)
    
    return coords_withoutDuplicates

def writeMesh(coords,filename):
    with open(filename, 'wb') as f:
        np.save(f, coords, allow_pickle=True)
        
def readMesh(filename):
    with open(filename, 'rb') as f:
        coords = np.load(f)
    return coords

def writeBoundaryNodes(nodes,filename):
    with open(filename, 'wb') as f:
        np.save(f, nodes, allow_pickle=True)
        
def readBoundaryNodes(filename):
    with open(filename, 'rb') as f:
        nodes = np.load(f)
    return nodes
        
def findNodesInsideGeom(coords, geomBoxExtension):
    '''
    Checking if the reference mesh nodes lie within the geom. The geom is square for now
    '''
    for inode in range(coords.shape[0]):
        if abs(coords[inode][0])  < geomBoxExtension and abs(coords[inode][1])  < geomBoxExtension:
            print ('node no.',inode,'with coords',coords[inode][0],coords[inode][1],'is inside geom')
    return

def plotTwoFields(field1,coords1,field2,coords2):
    '''
    Plots contours for two fields on same or different meshes
    This functions is specifically made for rectangular case as it uses it boundaries and scales
    '''
    # finding the max value of the field
    val_max = max(np.max(field1),np.max(field2))
    val_min = min(np.min(field1),np.min(field2))
    # val_min = 0
    step = 0.1
    print ('val_max: ', val_max,' and val_min:',val_min)
    # plotting and comparing the interpolated array
    fig, (ax1,ax2) = plt.subplots(ncols=2)  
    levels = np.arange(val_min, val_max, step)
    # ax1.tricontour(coords1[:,0], coords1[:,1], field1[:,0], levels=15, linewidths=0.1, colors='k')
    cntr1 = ax1.tricontourf(coords1[:,0], coords1[:,1], field1[:,0], levels=levels, cmap="RdBu_r")
    fig.colorbar(cntr1, ax=ax1)
    # # Add the patch to the Axes
    # rect = patches.Rectangle((-2, -2), 4, 4, linewidth=1, edgecolor='black', facecolor='none')
    # ax1.add_patch(rect)
    # ax1.set(xlim=(-50, 50), ylim=(-50, 50))

    # ax2.tricontour(coords2[:,0], coords2[:,1], field2[:,0], levels=15, linewidths=0.1, colors='k')
    cntr2 = ax2.tricontourf(coords2[:,0], coords2[:,1], field2[:,0], levels=levels, cmap="RdBu_r")
    fig.colorbar(cntr2, ax=ax2)
    # # Add the patch to the Axes
    # rect = patches.Rectangle((-2, -2), 4, 4, linewidth=1, edgecolor='black', facecolor='none')
    # ax2.add_patch(rect)
    # ax2.set(xlim=(-50, 50), ylim=(-50, 50))
    
    plt.show()
    return

def subtractMean(array,mean):
    aux = np.copy(array)
    array = np.subtract(aux,mean)
    return array

def addMean(array,mean):
    aux = np.copy(array)
    array = np.add(aux,mean)
    return array

def rotate_point(point, angle, center_point=(0, 0)):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = np.radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * np.cos(angle_rad) - new_point[1] * np.sin(angle_rad),
                 new_point[0] * np.sin(angle_rad) + new_point[1] * np.cos(angle_rad))
    # Reverse the shifting we have done
    new_point = (new_point[0] + center_point[0], new_point[1] + center_point[1])
    return new_point

def rotate_Mesh(coords,angle, center_point=(0, 0)):
    """Rotates a the mesh points around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    for ipoin in range(coords.shape[0]):
        point = (coords[ipoin,0],coords[ipoin,1])
        coords[ipoin,:] = rotate_point(point,angle,center_point=(0, 0))
    return coords

def writeBasis(basis,filename):
    '''
    Writes basis in filename in HDF5 format
    '''
    f1 = h5py.File(filename, 'a')
    for icol in range(basis.shape[1]):
        f1.create_dataset(f'basis/{icol}',data=basis[:,icol])
    return

def writeMean(mean,filename):
    '''
    Writes mean for snpashots in filename in HDF5 format
    '''
    f1 = h5py.File(filename, 'a')
    f1.create_dataset(f'mean',data=mean)
    return

def writeEigenValues(sigma,filename):
    '''
    Writes eigenValues for snpashots in filename in HDF5 format
    '''
    f1 = h5py.File(filename, 'a')
    f1.create_dataset(f'eigenValues',data=sigma)
    return

def writeCoefs(coefs,filename):
    with open(filename, 'wb') as f:
        np.save(f, coefs, allow_pickle=True)
        
def readBasis(nbasis,filename):
    '''
    Read basis in filename in HDF5 format
    '''
    f1 = h5py.File(filename, 'r')
    for icol in range(nbasis):
        basis_aux = np.array(f1.get(f'basis/{icol}'))
        if icol == 0:
            basis = basis_aux
        else:
            basis = np.column_stack((basis,basis_aux))
    return basis

def readMean(filename):
    '''
    Read mean for snpashots in filename in HDF5 format
    '''
    f1 = h5py.File(filename, 'r')
    mean = np.array(f1.get('mean'))
    return mean

def readEigenValues(filename):
    '''
    Read eigenValues for snpashots in filename in HDF5 format
    '''
    f1 = h5py.File(filename, 'r')
    sigma = np.array(f1.get(f'eigenValues'))
    return sigma   

def readCoef(filename,dofr=0):
    with open(filename, 'rb') as f:
        coefs_aux = np.load(f)
        if dofr ==0: 
            coefs = coefs_aux.copy()
        else:
            coefs = coefs_aux[:dofr,:].copy()
    return coefs

def calculateROMDofs(s_aux,ric):
    '''
    Calculates modes for ROM based on ric
    '''    
    energy = 0.0
    idofr = 0
    energy_total = sum(s_aux)
    energy_req = energy_total*ric
    while energy < energy_req:
        energy = energy + s_aux[idofr]
        idofr = idofr+1 
    dofr = idofr
    return dofr
#%% MAIN FUNCTIONS

def prepareData(case):
    field = case.field                                            # fields available: U, P
    angle_new = case.angle_new
    grados = case.grados                         # angles available
    fillValue = case.fillValue                                       # random high value to fill-in my mesh interpolator
    mesh_interpolation = case.mesh_interpolation                           # interpolation to be used by griddata
    npoins = case.npoins                                           # no of points for created Structured reference mesh
    rotateMesh = case.rotateMesh
    useRefMeshForProjection = case.useRefMeshForProjection
    root_data = fr'{case.root_data}\{case.folder_training}'  
    file_refMesh = case.file_refMesh
    file_boun = case.file_boun
    
    # Delete previous results
    for filename in glob.glob(fr'{root_data}\*_mode_*.vtk'):
       os.remove(filename)
    for filename in glob.glob(fr'{root_data}\*_reference_*.vtk'):
       os.remove(filename)
    for filename in glob.glob(fr'{root_data}\*_angle_*.vtk'):
       os.remove(filename)  
    for filename in glob.glob(fr'{root_data}\basis'):
       os.remove(filename)
    for filename in glob.glob(fr'{root_data}\mesh.txt'):
        os.remove(filename)
    for filename in glob.glob(fr'{root_data}\coefs.txt'):
        os.remove(filename)
         
    # Copy the mesh file 0 degree and unknown angle
    angles_forFieldRemoval = [0,angle_new]
    for iangle in angles_forFieldRemoval:
        src = fr'{root_data}\{field}_surfaces_{iangle}.vtk'
        dst = fr'{root_data}\{field}_reference_{iangle}.vtk'
        shutil.copy(src, dst)
        # file = "{}_reference_{}.vtk".format(field, angle_new)
    
    if useRefMeshForProjection:
        # Create the Reference Mesh
        # read a case mesh to find the boundaries of the domain for reference mesh
        _, coords_ref = f_vtk.files_vtk_to_array(root_data, [f'{field}_reference_0'],getCoords=True)   # case of 0 degree
        # create the structured mesh
        x_max = max(coords_ref[:,0])
        x_min = min(coords_ref[:,0])
        y_max = max(coords_ref[:,1])
        y_min = min(coords_ref[:,1])
        radius = max(abs(x_max),abs(x_min), abs(y_max), abs(y_min))                    # setting the radius to be the max extent
        coords_refMesh = createStructuredMesh(radius,npoins,shape='circular')
        coords_ref = coords_refMesh
        # save mesh
        writeMesh(coords_ref,fr'{root_data}\{file_refMesh}')
        
        # Calculating mesh projection error for 0 or angle_new degree case
        angle_forMeshProjectionTest = angle_new
        field_true, coords_test = f_vtk.files_vtk_to_array(root_data, [f'{field}_surfaces_{angle_forMeshProjectionTest}'],getCoords=True)
        # find boundary nodes to enforce noSlip after projection   
        boundaryNodes = np.where(field_true < 1e-10)[0]
        field_true_interpolated = projectFieldToNewMesh(coords_test, field_true, coords_ref, mesh_interpolation ,fillValue, rescale=True)
        # plotTwoFields(field_true,coords_test,field_true_interpolated,coords_ref)
        field_true_projected = projectFieldToNewMesh(coords_ref, field_true_interpolated, coords_test, mesh_interpolation ,fillValue, rescale=True)
        # correct noSlip BCs voilated due to interpolation
        field_true_projected[boundaryNodes] = 0.0
        # write boundary nodes
        writeBoundaryNodes(boundaryNodes, fr'{root_data}\{file_boun}')
        
        dir_new_file = f_vtk.array_to_file_vtk(root_data, f'{field}_reference_{angle_forMeshProjectionTest}', f'{field}_angle_{angle_forMeshProjectionTest}_meshProjection',field_true_projected, field)
        diff_fieldMeshProjected = np.subtract(field_true[:,0],field_true_projected[:,0])
        diff_fieldMeshProjected_norm = np.linalg.norm(diff_fieldMeshProjected,2)/np.linalg.norm(field_true,2)
        # plotTwoFields(field_true,coords_test,field_true_projected,coords_test)
        
        if rotateMesh:
            # Calculating mesh rotation+projection error for angle_new degree case
            angle_forMeshProjectionTest = angle_new
            field_true, coords_test = f_vtk.files_vtk_to_array(root_data, [f'{field}_surfaces_{angle_forMeshProjectionTest}'],getCoords=True)
            coords_test = rotate_Mesh(coords_test, angle_forMeshProjectionTest)
            field_true_interpolated = projectFieldToNewMesh(coords_test, field_true, coords_ref, mesh_interpolation ,fillValue, rescale=True)
            # plotTwoFields(field_true,coords_test,field_true_interpolated,coords_ref)
            field_true_projected = projectFieldToNewMesh(coords_ref, field_true_interpolated, coords_test, mesh_interpolation ,fillValue, rescale=True)
            # correct noSlip BCs voilated due to interpolation
            field_true_projected[boundaryNodes] = 0.0
            
            coords_test = rotate_Mesh(coords_test, -angle_forMeshProjectionTest)
            dir_new_file = f_vtk.array_to_file_vtk(root_data, f'{field}_reference_{angle_forMeshProjectionTest}', f'{field}_angle_{angle_forMeshProjectionTest}_meshProjection',field_true_projected, field)
            diff_fieldMeshProjectedAndRotated = np.subtract(field_true[:,0],field_true_projected[:,0])
            diff_fieldMeshProjectedAndRotated_norm = np.linalg.norm(diff_fieldMeshProjected,2)/np.linalg.norm(field_true,2)
            # plotTwoFields(field_true,coords_test,field_true_projected,coords_test)
    
    # Read the training data and intepolate if needed
    files = ["{}_surfaces_{}".format(field, grado) for grado in grados]
    iarray =0
    data_training =[]
    for ifile in files:
        array, coords = f_vtk.files_vtk_to_array(root_data, [ifile],getCoords=True)
        array_backup = array.copy()
        coords_backup = coords.copy()
        if rotateMesh and grados[iarray] != 0:
            coords = rotate_Mesh(coords, grados[iarray])
            # plotTwoFields(array,coords_backup,array,coords)
        if useRefMeshForProjection:
            array = projectFieldToNewMesh(coords, array, coords_ref, mesh_interpolation ,fillValue, rescale=True)              
            # plotTwoFields(array_backup,coords,array,coords_ref)
        if iarray == 0:
            data_training = array
        else:
            data_training = np.append(data_training,array,axis=1)
        iarray = iarray +1 

    return data_training

def trainModel(case,data_training):    
    field = case.field                                            # fields available: U, P
    use_SVD = case.use_SVD                                         # perform SVD or not
    subtract_mean = case.subtract_mean                                   # subtract mean for SVD
    root_data = fr'{case.root_data}\{case.folder_training}'  
    file_basis = case.file_basis
    file_coefs = case.file_coefs
    
    # Perform SVD
    if use_SVD:    
        # subtract mean from training data
        if subtract_mean:
            data_training_mean = np.mean(data_training,axis=1)
            for icol in range (data_training.shape[1]):
                data_training[:,icol] = subtractMean(data_training[:,icol], data_training_mean)
        # descomposición svd
        u, s, vt = np.linalg.svd(data_training, full_matrices=False)
        
        # writing basis, mean and sigular values
        file_basis = fr'{root_data}\{file_basis}'
        writeBasis(u, file_basis)
        writeEigenValues(s, file_basis)
        if subtract_mean: writeMean(data_training_mean,file_basis)
        
        # writing the POD modes for visualization
        imode = 0
        for u_column in range(u.shape[1]):
            imode = imode+1
            new_file = "{}_mode_{}".format(field, u_column+1)
            # dir_new_file = f_vtk.array_to_file_vtk(root_data,f'{field}_reference_0', new_file, u[:, u_column], field)
    
        # finding the coeffs for training data
        A_coef = np.matmul(np.transpose(u[:,:]), data_training)    # coefficient matrix 
        
        # write coeffs to a file
        writeCoefs(A_coef, fr'{root_data}\{file_coefs}')
    return 

def evaluateModel(case,wind_speed, wind_dir,data_training = None, scale_velocity=True,compare_results = False):     
    # User inputs
    field = case.field                                            # fields available: U, P
    angle_new = wind_dir
    wind_ref = case.wind_ref
    use_SVD = case.use_SVD                                         # perform SVD or not
    subtract_mean = case.subtract_mean                                   # subtract mean for SVD
    interp_method = case.interp_method                                  # interp_quad, lagrange,Rbf
    grados = case.grados                                    # angles available
    fillValue = case.fillValue                                       # random high value to fill-in my mesh interpolator
    ric = case.ric                                             # information content to retain in the POD modes 
    mesh_interpolation = case.mesh_interpolation                           # interpolation to be used by griddata
    rotateMesh = case.rotateMesh
    useRefMeshForProjection = case.useRefMeshForProjection
    root_data = fr'{case.root_data}\{case.folder_evaluation}'  
    file_test = case.file_test 
    file_refMesh = case.file_refMesh
    file_basis = case.file_basis
    file_coefs = case.file_coefs
    file_boun = case.file_boun

    if use_SVD:
        s = readEigenValues(fr'{root_data}\{file_basis}')
        # finding the no. of modes to retain
        energy = 0.0
        idofr = 1
        energy_total = sum(s)
        energy_req = energy_total*ric
        while energy < energy_req:
            energy = energy + s[idofr-1]
            idofr = idofr+1 
        dofr = idofr-1
        
        # read data
        u = readBasis(dofr, fr'{root_data}\{file_basis}')
        A_coef = readCoef(fr'{root_data}\{file_coefs}',dofr=dofr)                   
        if subtractMean: data_training_mean = readMean(fr'{root_data}\{file_basis}')
        if useRefMeshForProjection: 
            coords_ref = readMesh(fr'{root_data}\{file_refMesh}')
            boundaryNodes = readBoundaryNodes(fr'{root_data}\{file_boun}')

        # # Plotting coefficients    
        # isample = np.arange(1,A_coef.shape[0]+1)
        # for i in range(1,A_coef.shape[0]):
        #     plt.plot (isample,A_coef[i,:], label=i+1)
        # plt.legend()
        # plt.show()
    
        # # Vamos a ver la importancia de los distintos modos    
        # plt_lib.plot_cum_exp(s)
        
        # interpolation to find the new coefficients
        new_coef = np.zeros([ A_coef.shape[0],1])
        for i in range(A_coef.shape[0]):
            if interp_method == 'interp_quad':
                # using interp, not sure which kind of interpolation is used here
                f = interpolate.interp1d(np.radians(grados), A_coef[i, :], kind = 'quadratic')
            elif interp_method == 'lagrange':    
                # lagrange interpolation
                f = lagrange(np.radians(grados), A_coef[i, :])   
            elif interp_method == 'Rbf': 
                # Radial Basis Fuction
                f = Rbf(np.radians(grados), A_coef[i, :])
            new_coef[i,0] = f(np.radians(angle_new))
        # obtenemos el nuevo campo:
        field_new = np.matmul(u[:,:dofr], np.array(new_coef).reshape(-1, 1)) 
        if subtract_mean:
            field_new[:,0] = addMean(field_new[:,0], data_training_mean)    #add mean
    
        # comparing basis projection error, this is the minimum error ROM can achieve
        if compare_results: 
            field_true, coords_test = f_vtk.files_vtk_to_array(root_data, file_test,getCoords=True)
            field_backup = field_true.copy()
            coords_test_backup = coords_test.copy()
            if rotateMesh:
                coords_test = rotate_Mesh(coords_test, angle_new)
            if useRefMeshForProjection:
                field_true = projectFieldToNewMesh(coords_test, field_true, coords_ref, mesh_interpolation ,fillValue, rescale=True)
            if subtract_mean:
                field_true[:,0] = subtractMean(field_true[:,0], data_training_mean) 
            actual_coef = np.matmul(np.transpose(u[:,:dofr]), field_true)   
            field_true_recreated = np.matmul(u[:,:dofr], actual_coef)
            if subtract_mean:
                field_true_recreated[:,0] = addMean(field_true_recreated[:,0], data_training_mean) 
            if useRefMeshForProjection:
                field_true_recreated = projectFieldToNewMesh(coords_ref, field_true_recreated, coords_test, mesh_interpolation ,fillValue, rescale=True)
                # correct noSlip BCs voilated due to interpolation
                field_true_recreated[boundaryNodes] = 0.0
            if rotateMesh:
                coords_test = rotate_Mesh(coords_test, -angle_new)
            dir_new_file = f_vtk.array_to_file_vtk(root_data,f'{field}_reference_{angle_new}' ,f'{field}_angle_{angle_new}_romRecreated',field_true_recreated, field)
            diff_fieldRecreated = np.subtract(field_backup[:,0],field_true_recreated[:,0])
            diff_fieldRecreated_norm = np.linalg.norm(diff_fieldRecreated,2)/np.linalg.norm(field_backup,2)
            diff_coeffRecreated = np.subtract(actual_coef[:,0],new_coef[:,0])
            diff_coeffRecreated_norm = np.linalg.norm(diff_coeffRecreated,2)/np.linalg.norm(actual_coef,2)
            # plotTwoFields(field_backup,coords_test_backup,field_true_recreated,coords_test)
    
    # Perform Direct Mesh Interpolation
    else:
        # Interpolation to find the new field
        field_new = np.zeros([data_training.shape[0],1])
        for i in range(data_training.shape[0]):
            if interp_method == 'interp_quad':
                # using interp, not sure which kind of interpolation is used here
                f = interpolate.interp1d(np.radians(grados), data_training[i, :], kind = 'quadratic')
            elif interp_method == 'lagrange':    
                # lagrange interpolation
                f = lagrange(np.radians(grados), data_training[i, :])   
            elif interp_method == 'Rbf': 
                # Radial Basis Fuction
                f = Rbf(np.radians(grados), data_training[i, :])
            field_new[i,0] = f(np.radians(angle_new)) 
    
    if useRefMeshForProjection:    
        # Project back the field to its original mesh     
        _, coords_new = f_vtk.files_vtk_to_array(root_data, [f'{field}_surfaces_{angle_new}'],getCoords=True)
        coords_new = rotate_Mesh(coords_new, angle_new)
        field_new = projectFieldToNewMesh(coords_ref, field_new, coords_new, mesh_interpolation ,fillValue, rescale=True)
        # correct noSlip BCs voilated due to interpolation
        field_new[boundaryNodes] = 0.0
    
    if rotateMesh:
        # Rotate back the field as per its angle         
        coords_new = rotate_Mesh(coords_new, -angle_new)
        plotTwoFields(field_backup,coords_test_backup,field_new,coords_new)
        
    if compare_results:
        # Calculating field interpolation error
        field_true, coords_test = f_vtk.files_vtk_to_array(root_data, file_test,getCoords=True)
        diff_fieldInterp = np.subtract(field_true[:,0],field_new[:,0])
        diff_fieldInterp_norm = np.linalg.norm(diff_fieldInterp,2)/np.linalg.norm(field_true,2)
            
        # Plotting original and interpolated fields
        field_true, coords_test = f_vtk.files_vtk_to_array(root_data, file_test,getCoords=True)
        plotTwoFields(field_true,coords_test,field_new,coords_test)
    
    # Scale as per actual wind speed
    if scale_velocity: field_new = field_new * (wind_speed/wind_ref)
    # Vamos a crear el nuevo fichero vtk con el nuevo campo a angle_new grados
    dir_new_file_rotated = f_vtk.array_to_file_vtk(root_data, f'{field}_reference_{angle_new}',f'{field}_angle_{angle_new}_interp', field_new, field)  

    return field_new

#%% MAIN
if __name__ == "__main__":
    # Import functions and classes
    from def_classes import case
    
    # User inputs
    inputs = case()
    inputs.field = "U"                                            # fields available: U, P
    inputs.angle_new = 65
    inputs.use_SVD = True                                         # perform SVD or not
    inputs.subtract_mean = True                                   # subtract mean for SVD
    inputs.interp_method = 'Rbf'                                  # interp_quad, lagrange,Rbf
    inputs.grados = [degree_div_10*10 for degree_div_10 in list(range(37))]                         # angles available
    # inputs.grados = [degree_div_10*10 for degree_div_10 in list(range(5,9))]                         # angles available
    inputs.ric = 1.00                                             # information content to retain in the POD modes 
    inputs.mesh_interpolation = 'linear'                           # interpolation to be used by griddata
    inputs.npoins = 1000                                           # no of points for created Structured reference mesh
    inputs.rotateMesh = False
    inputs.useRefMeshForProjection = False
    inputs.folder_training = 'training_urbanExample'
    inputs.folder_evaluation = 'evaluation_urbanExample'
    inputs.month = 9                                              
    inputs.day = 3
    inputs.hour = 18
    
    inputs.root_data = r"C:\Zulkee\RDT\RDTSimulations\project_buildingMachineLearning"  
    inputs.file_test = ["{}_surfaces_{}".format(inputs.field, str(inputs.angle_new))] 
    
    # Calculations
    # data = prepareData(inputs)
    # trainModel(inputs, data)
    field = evaluateModel(inputs, wind_speed= 5, wind_dir=inputs.angle_new, scale_velocity=False, compare_results = True)