# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:20:16 2023

@author: Workstation5
"""

# -*- coding: utf-8 -*-
import sys
import pandas as pd
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

''' Este fichero contiene funciones útiles para lectura, escritura y modificaciones en general
    de archivos vtk'''


def poldata_comp_vect_mag(polydata, field):
    """
    function:  poldata_comp_vect_mag

    Descripción: Esta función recibe datos de tipo polydata con campo vectorial y calcula el
                 módulo en cada punto.

    Input:

        polydata: polydata con malla y field vectorial. (Velocidades)
        magnitudes:

    Output:
        magnitudes: La función escribe como salida un array con los valores de la magnitud.

    """

    # Veamos si podemos acceder a los datos en el polydata
    try:
        point_data = polydata.GetPointData()
    except:
        print("Hay un problema con polydata, checkea")
        sys.exit(1)

    # ahora si podemos acceder al field (U por ejemplo) del polydata
    try:
        vectors = point_data.GetArray(field)
    except:
        print("No se encuentra el field proporcionado, checkea los campos")
        sys.exit(1)

    # veamos cuántos puntos hay en la malla
    num_points = polydata.GetNumberOfPoints()

    # Vamos a crear un array vacío donde poner las magnitudes
    if num_points > 0:
        magnitudes = np.zeros(num_points)
    else:
        print("No se han encontrado puntos, checkea el polydata")
        sys.exit(1)

    # Para cada índice de la malla, calculemos y guardemos el módulo del campo vectorial:
    for i in range(num_points):
        v = np.array(vectors.GetTuple3(i))
        print("v{}: ".format(i), v, "\n")
        magnitudes[i] = np.linalg.norm(v)
        print("magnitudes[{}]: ".format(i), magnitudes[i], "\n")
    return magnitudes


def wr_poldata_vector_to_scal(polydata, magnitudes, output_filename):
    """
    function: wr_poldata_vector_to_scal

    Descripción: Esta función toma un fichero vtk con campos vectoriales y estructura polydata
                 y crea un nuevo fichero vtk donde el campo es escalar y cuyo valores en la
                 estructura polydata son iguales al módulo del campo vectorial.

    Input:

        polydata: polydata con malla y field vectorial. (Velocidades)
        magnitudes: array donde guardamos las magnitudes que queremos para la estructura del
                    polydata.
        output_filename: dirección del fichero donde se va a guardar el nuevo fichero vtk.

    Output:
        scalarField_array: La función escribe como salida a este array.

    """

    # Crear un objeto vtkFloatArray con los valores de magnitudes
    scalar_array = vtk.vtkFloatArray()
    scalar_array.SetName("Magnitudes")
    scalar_array.SetNumberOfComponents(1)
    scalar_array.SetNumberOfTuples(polydata.GetNumberOfPoints())
    for i in range(polydata.GetNumberOfPoints()):
        scalar_array.SetValue(i, magnitudes[i])

    # Crear un nuevo objeto vtkPolyData y copiar la estructura de la original
    new_polydata = vtk.vtkPolyData()
    new_polydata.DeepCopy(polydata)

    # Asignar el campo escalar de magnitudes al nuevo objeto vtkPolyData
    new_polydata.GetPointData().SetScalars(scalar_array)

    # Escribir el nuevo fichero vtk
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(output_filename)
    writer.SetInputData(new_polydata)
    writer.Write()


# *************************************** LECTURA Y TRANSFORMACIÓN DATOS **********************************************


def files_vtk_to_array(root_data, files):
    """
    function: files_vtk_to_array

    Descripción: Esta función transforma una serie de ficheros vtk que contiene fields como p, T y v y los
    acopla en un array donde cada columna corresponde a un fichero vtk.

    Input:

        files: Lista que contiene los nombres de los ficheros vtk a convertir.
        root_data: Lugar donde se encuentran los ficheros vtk.

    Output:
        scalarField_array: La función escribe como salida a este array.

    """

    '''
    Recorrido entre los ficheros vtk:
    Este bloque es utilizado para abrir cada uno de los casos y acoplar sus fields a un array
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
            # Obtenemos el field T
            # scalarField_pdata = point_data.GetArray("U")
            # Y lo convertimos a un array
            # scalarField_vector = vtk_to_numpy(scalarField_pdata)

            scalarField_vector = poldata_comp_vect_mag(polydata, "U")
            array_to_file_vtk(root_data, scalarField_vector, file, "U")
        # Para la primera iteracion
        if count == 0:
            # Vemos cuantos filas tiene cada field
            # n_rows = len(scalarField_vector)
            n_rows = polydata.GetNumberOfPoints()
            # Creamos un array con las mismas filas y vacío
            scalarField_array = np.empty([n_rows, 0])

        # Añadimos el array vacio al array creado en cada iteracion:
        scalarField_array = np.append(scalarField_array, scalarField_vector[:, np.newaxis], axis=1)

    return scalarField_array


# ************************** RECONSTRUCTION FICHERO VTK CON EL NUEVO FIELD *****************************

def array_to_file_vtk(root_data, new_array, new_file, field):
    """

    function: array_to_file_vtk

    Descripción: Esta función convierte un array columna a un field de vtk y lo añade a un fichero de referencia.

    Input:

        root_data: Lugar donde se encuentra el fichero vtk de referencia
        new_array: Nuevo array para definir en el field.
        new_file:  Nombre que tomará el nuevo fichero a guardar con el new_array

    Output:

        new_dir_file: Retorna la dirección completa del nuevo fichero vtk.

        Además, la función escribe un nuevo fichero vtk con los valores de new_array y el nombre de new_file.

    """

    print("Estamos en array_to_file_vtk\n")

    dir_file = os.path.join(root_data, "reference_{}.vtk".format(field))
    new_dir_file = os.path.join(root_data, new_file + ".vtk")

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
        '''
        # Obtenemos el field U
        scalarField_pdata = point_data.GetArray("U")
        # Y lo convertimos a un array
        scalarField_vector = vtk_to_numpy(scalarField_pdata)
        # Pasamos el nuevo array a nuestro campo
        scalarField_vector[:] = new_array[:]
        # Le hacemos saber a vtk que hemos hecho esa modificación para que lo tenga en cuenta:
        scalarField_pdata.Modified()
        '''

        wr_poldata_vector_to_scal(polydata, new_array, new_dir_file)

    return new_dir_file
