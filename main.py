# -*- coding: utf-8 -*-

import numpy as np
import func_vtk as f_vtk
import svd_interp as svd_op
import plotting_libr as plt_lib
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy

"""
Created on Fri Mar  3 20:22:02 2023

@author: Workstation5
"""

# Datos ficheros vtk

root_data = r"C:\Zulkee\RDT\RDTSimulations\project_buildingMachineLearning\v0\results"

field = "U"
grados = [0, 30, 60, 90]
files = ["{}_surfaces_{}".format(field, grado) for grado in grados]

# 1. Pasemos los ficheros de presiones, temperaturas o velocidades vtk a arrays:

np_escalar = f_vtk.files_vtk_to_array(root_data, files)

# 2. Descomposición svd

# u, smat, vt = svd_op.svd_array(np_escalar)
u, s, vt = np.linalg.svd(np_escalar)

# Vamos a ver la importancia de los distintos modos

plt_lib.plot_cum_exp(s)

# Interpolación a un nuevo modo. En este caso a 45 grados usando los modos de 30 y 60 grados:
x = [30, 60]
x_new = 45

vt_new_mode = svd_op.interp_lin(x, vt, x_new)

# Obtenemos el nuevo campo de presiones:

n_samples = np_escalar.shape[0]
n_cols = np_escalar.shape[1]

field_new = np.zeros(n_samples)

for i in range(n_cols):
    field_new[:] = field_new[:] + s[i] * vt_new_mode[i] * u[:, i]

# Vamos a crear el nuevo fichero vtk con las nuevas presiones a 45 grados

grado = 45
new_file = "{}_angle_{}_interp".format(field, grado)

dir_new_file = f_vtk.array_to_file_vtk(root_data, field_new, new_file, field)

# Ploteamos el fichero vtk correspondiente a 45 grados

# plt_lib.plot_vtk_byVTK(dir_new_file)

# plt_lib.plot_vtk_byMaYAVI(dir_new_file)

