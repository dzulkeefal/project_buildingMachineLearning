import matplotlib.pyplot as plt
import func_aux as f_aux
import numpy as np
# import mayavi.mlab as mlab
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os


def plot_modes(n_modes, n_rows_fig, U, s, VT):
    n_cols_fig = int(n_modes / n_rows_fig)
    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    for i in range(0, n_modes):
        try:
            mat_i = s[i] * U[:, i].reshape(-1, 1) @ VT[i, :].reshape(1, -1)
            # mat_i = U[:, i] @ s[i, i] @ VT[i, :]
        except:
            print("El error está en el producto")

        axes[i // n_cols_fig, i % n_cols_fig].imshow(mat_i)
        axes[i // n_cols_fig, i % n_cols_fig].set_title("$\sigma_{0}\mathbf{{u_{0}}}\mathbf{{v_{0}}}^T$".format(i + 1),
                                                        fontsize=16)

    plt.show()


def plot_cum_exp(s):
    '''
    plot_cum: Función que puede ser usada para plotear el cumulativo y explicada de un 1D array

    input:
        s: array 1D
    output:
        ploteo del cumulativo y el explicado de s
    '''

    # Calculamos el total de los elementos del array
    tot_s = sum(s)

    # Creamos una lista que contenga los valores relativos al total
    s_exp = [i / tot_s for i in sorted(s, reverse=True)]

    # Calculamos la lista con los valores acumulativos
    cum_s_exp = np.cumsum(s_exp)

    # Creamos la lista para tener en cuenta el índice del 1D array
    list_nModes = [i for i in range(len(s))]

    # Ploteamos lo calculado
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.bar(list_nModes, s_exp)
    ax.step(list_nModes, cum_s_exp)

    plt.show()


# def plot_vtk_byMaYAVI(dir_vtkFile):
#     '''
#     plot_vtk: Función que puede ser usada para plotear un archivo vtk de un field (presiones, etc.)

#     input:
#         dir_vtkFile: dirección del fichero vtk a plotear
#     output:
#         ploteo del archivo vtk
#     '''
#     Cargamos el archivo vtk
#     reader = mlab.pipeline.open(dir_vtkFile)

#     Visualizar datos
#     mlab.pipeline.surface(reader)
#     mlab.show()


def plot_vtk_byVTK(dir_vtkFile):
    # Abrimos el actual fichero .vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(dir_vtkFile)
    reader.Update()

    polydata = reader.GetOutput()

    point_data = polydata.GetPointData()
    points = polydata.GetPoints()
    points_np = vtk_to_numpy(points.GetData())

    if point_data.HasArray("p"):

        scalarField_pdata = point_data.GetArray("p")
        scalarField_vector = vtk_to_numpy(scalarField_pdata)

    elif point_data.HasArray("T"):

        scalarField_pdata = point_data.GetArray("T")
        scalarField_vector = vtk_to_numpy(scalarField_pdata)

    # Graficar datos
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='rectilinear')
    my_plot = ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=scalarField_vector, cmap='jet')
    cbar = plt.colorbar(my_plot)
    # Personalizar etiquetas de la barra de valores
    cbar.set_label('Valores de color')
    plt.show()


def plot_modes_vtk(n_columns, root_data, name_files):
    fig, axes = plt.subplots(1, n_columns, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3, hspace=0.2)

    for i, name_file in enumerate(name_files):
        try:
            dir_file = os.path.join(root_data, name_file + ".vtk")
            points_np, scalarField_vector = f_aux.vtk_extract_field_points(dir_file)
        except:
            print("El error está en el producto")

        axes[i].scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=scalarField_vector, cmap='jet')

    plt.show()
