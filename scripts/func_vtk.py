# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:20:16 2023

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
import pandas as pd
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

''' Este fichero contiene funciones útiles para lectura, escritura y modificaciones en general
    de archivos vtk'''


#%%  FUNCTIONS
def poldata_comp_vect_mag(polydata, field):
    """
    function:  poldata_comp_vect_mag

    Descripción: This function receives data of type polydata with vector field and calculates the
                module at each point.

    Input:

        polydata: polydata with mesh and vector field. (Speeds)
         magnitudes:

    Output:
        magnitudes: The function writes as output an array with the values of the magnitude.

    """

    # Let's see if we can access the data in the polydata
    try:
        point_data = polydata.GetPointData()
    except:
        print("Hay un problema con polydata, checkea")
        sys.exit(1)

    # ahora si podemos acceder al field (U por ejemplo) del polydata
    try:
        vectors = point_data.GetArray(field)
    except:
        print("The provided field is not found, check the fields")
        sys.exit(1)

    # Now we can access the field (U for example) of the polydata
    num_points = polydata.GetNumberOfPoints()

    # We are going to create an empty array where we can put the magnitudes
    if num_points > 0:
        magnitudes = np.zeros(num_points)
    else:
        print("No points found, check the polydata")
        sys.exit(1)

    #For each index of the mesh, let's calculate and save the modulus of the vector field:
    for i in range(num_points):
        v = np.array(vectors.GetTuple3(i))
        #print("v{}: ".format(i), v, "\n")
        magnitudes[i] = np.linalg.norm(v)
        #print("magnitudes[{}]: ".format(i), magnitudes[i], "\n")
    return magnitudes


def wr_poldata_vector_to_scal(polydata, magnitudes, output_filename, field):
    """
    function: wr_poldata_vector_to_scal

    Description: This function takes a vtk file with vector fields and polydata structure
                  and creates a new vtk file where the field is scalar and whose values in the
                  polydata structure are equal to the modulus of the vector field.

     Input:

         polydata: polydata with mesh and vector field. (Speeds)
         magnitudes: array where we store the magnitudes that we want for the structure of the
                     polydata.
         output_filename: address of the file where the new vtk file will be saved.

     Output:
         scalarField_array: The function writes this array as output.

    """

    #Create a vtkFloatArray object with the magnitude values

    scalar_array = vtk.vtkFloatArray()
    scalar_array.SetName("Magnitud_velocity")
    scalar_array.SetNumberOfComponents(1)
    scalar_array.SetNumberOfTuples(polydata.GetNumberOfPoints())
    for i in range(polydata.GetNumberOfPoints()):
        scalar_array.SetValue(i, magnitudes[i])


    # Create a vtkFloatArray object with the magnitude values
    new_polydata = vtk.vtkPolyData()
    new_polydata.DeepCopy(polydata)
    # Delete existing vector field
    new_polydata.GetPointData().RemoveArray(field)
    # Assign the magnitude scalar field to the new vtkPolyData object
    new_polydata.GetPointData().SetScalars(scalar_array)

    # Write the new vtk file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(new_polydata)
    writer.Write()


# *************************************** READING AND TRANSFORMATION OF DATA**********************************************


def files_vtk_to_array(root_data, files, getCoords = False):
    """
    function: files_vtk_to_array

    Description: This function transforms a series of vtk files that contain fields such as p, T and v and the
     dock in an array where each column corresponds to a vtk file.

     Input:

         files: List containing the names of the vtk files to convert.
         root_data: Place where the vtk files are located.

     Output:
         scalarField_array: The function writes this array as output.

    """

    '''
    Tour between vtk files:
    This block is used to open each of the cases and attach their fields to an array
    '''
    print("Estamos en files_vtk_to_array\n")

    for count, file in enumerate(files):  # donde file es cada uno de los ficheros vtk.

        print("Leyendo el fichero {}: ".format(file))

        # Creamos la dirección completa del actual fichero .vtk
        dir_file = os.path.join(root_data, file + ".vtk")

        # Creamos un lector del actual fichero .vtk
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(dir_file)
        reader.Update()
        # Obtenemos los valores dentro que pueden ser tanto los fields como la malla.
        polydata = reader.GetOutput()
        # dim = polydata.GetDimensions()
        # Obtenemos los valores del field
        point_data = polydata.GetPointData()
        # Si tenemos el array p hacemos lo siguiente:
        if point_data.HasArray("p"):
            # Obtenemos el field p
            scalarField_pdata = point_data.GetArray("p")
            # Y lo convertimos a un array
            scalarField_vector = vtk_to_numpy(scalarField_pdata)
        # Si tenemos el array T hacemos lo siguiente:
        elif point_data.HasArray("T"):
            # Obtenemos el field T
            scalarField_pdata = point_data.GetArray("T")
            # Y lo convertimos a un array
            scalarField_vector = vtk_to_numpy(scalarField_pdata)

        elif point_data.HasArray("U"):

            scalarField_vector = poldata_comp_vect_mag(polydata, "U")
            # array_to_file_vtk(root_data, scalarField_vector, file, "U")
        # Para la primera iteracion
        if count == 0:
            # Vemos cuantos filas tiene cada field
            # n_rows = len(scalarField_vector)
            n_rows = polydata.GetNumberOfPoints()
            # Creamos un array con las mismas filas y vacío
            scalarField_array = np.empty([n_rows, 0])
            # get coordinates
            coords = np.zeros([n_rows,3])
            for i in range(n_rows):
                    coords[i,:] = polydata.GetPoint(i)
            

        # Añadimos el array vacio al array creado en cada iteracion:
        scalarField_array = np.append(scalarField_array, scalarField_vector[:, np.newaxis], axis=1)
    
    if getCoords == True:
        return scalarField_array, coords[:,0:-1]
    else:
        return scalarField_array


# ************************** RECONSTRUCTION FICHERO VTK CON EL NUEVO FIELD *****************************

def array_to_file_vtk(root_data, old_file, new_file,new_array,field):
    """

    function: array_to_file_vtk

    Descripción: Esta función convierte un array columna a un field de vtk y lo añade a un fichero de referencia.

    Input:

        root_data: Lugar donde se encuentra el fichero vtk de referencia
        new_array: Nuevo array para definir en el field.
        new_file:  Nombre que tomará el nuevo fichero a guardar con el new_array
        old_file: The file in which to write the new array    
    Output:

        new_dir_file: Retorna la dirección completa del nuevo fichero vtk.

        Además, la función escribe un nuevo fichero vtk con los valores de new_array y el nombre de new_file.

    """

    # print("Estamos en array_to_file_vtk\n")

    dir_file = os.path.join(root_data, f'{old_file}.vtk'.format(field))
    new_dir_file = os.path.join(root_data, f'{new_file}.vtk')

    # Creamos un lector del actual fichero .vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(dir_file)
    reader.Update()
    # Obtenemos los valores dentro que pueden ser tanto los fields como la malla.
    polydata = reader.GetOutput()
    # Obtenemos los valores del field
    point_data = polydata.GetPointData()

    # Si tenemos el array p hacemos lo siguiente:
    if point_data.HasArray("p"):
        # Obtenemos el field p
        scalarField_pdata = point_data.GetArray("p")
        # Y lo convertimos a un array
        scalarField_vector = vtk_to_numpy(scalarField_pdata)

        scalarField_vector[:] = new_array[:]

        scalarField_pdata.Modified()

        # Creamos un writer para crear un nuevo fichero vtk con el nuevo polydata
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(new_dir_file)
        writer.SetInputData(polydata)
        writer.Write()

    # Si tenemos el array T hacemos lo siguiente:
    elif point_data.HasArray("T"):
        # Obtenemos el field T
        scalarField_pdata = point_data.GetArray("T")
        # Y lo convertimos a un array
        scalarField_vector = vtk_to_numpy(scalarField_pdata)
        # Pasamos el nuevo array a nuestro campo
        scalarField_vector[:] = new_array[:]
        # Le hacemos saber a vtk que hemos hecho esa modificación para que lo tenga en cuenta:
        scalarField_pdata.Modified()

        # Creamos un writer para crear un nuevo fichero vtk con el nuevo polydata
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(new_dir_file)
        writer.SetInputData(polydata)
        writer.Write()

    # Si tenemos el array T hacemos lo siguiente:
    elif point_data.HasArray("U"):

        wr_poldata_vector_to_scal(polydata, new_array, new_dir_file, field)

    return new_dir_file

def plotFieldInParaview(case, field_new,wind_dir):
    root_data = fr'{case.root_data}\{case.folder_evaluation}'  
    field = case.field
    month = case.month
    day = case.day
    hour = case.hour
    file_new = f'{field}_{hour}_{day}_{month}_angle_{wind_dir}'
    array_to_file_vtk(root_data, f'{field}_reference_0',file_new, field_new, field)  
    return
# ************************** LIMPIEZA PARA FICHERO DE REFERENCIA *****************************

def delete_fields(root_data, file, field):
    """
    function: delete_fields

    Descripción: Esta función elimina los fields que se encuentren en un fichero vtk.

    Input:

        dir_vtk: dirección del fichero vtk donde guarda polydata con malla y field vectorial o escalar.

    Output:
        No hay salida, la función elimina solo los campos.
    """

    print("Leyendo el fichero {}: ".format(file))

    # Creamos la dirección completa del fichero actual .vtk
    dir_file = os.path.join(root_data, file)

    # Creamos un lector del actual fichero .vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(dir_file)
    reader.Update()
    # Obtenemos los valores dentro que pueden ser tanto los fields como la malla.
    polydata = reader.GetOutput()
    # Obtenemos los valores del field
    point_data = polydata.GetPointData()

    polydata.GetPointData().RemoveArray(field)
    print("Se ha eliminado el campo {}\n".format(field))

    scalar_array = vtk.vtkFloatArray()
    scalar_array.SetName(field)
    scalar_array.SetNumberOfComponents(1)
    scalar_array.SetNumberOfTuples(polydata.GetNumberOfPoints())
    point_data.AddArray(scalar_array)

    #for i in range(polydata.GetNumberOfPoints()):
    #    scalar_array.SetValue(i, magnitudes[i])

    # Crear el escritor del archivo VTK
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(dir_file)
    writer.SetInputData(polydata)
    writer.Write()
