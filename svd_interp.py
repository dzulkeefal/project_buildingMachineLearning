# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:43:08 2022

@author: syp_Franco
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intp

''' Este fichero está destinado unicamente a la lectura de los ficheros tvk
    que contienen información de simulaciones dependientes de parámetros.
    Para ello, se ha acordado leer info de ficheros vtk que contienen solo
    zonas específicas de simulación '''

# *************************************** SINGULAR VALUE DECOMPOSITION *************************************************

''' Reconstruir con menos modos de forma que los datos pesen mucho menos'''


def approx_array_nmodes(n_modes, u, s, vt):
    '''

    input:

        n_modes: Número de modos con lo que se desea interpolar

    output:

        u: Array de vectores singulares izquierdos
        s: Array valores singulares
        vt: Array de vectores singulares derechos

    '''

    smat = np.zeros((u.shape[0], vt.shape[0]))

    mat_approx = u[:, :n_modes] @ smat[:n_modes, :n_modes] @ vt[:n_modes, :]

    return mat_approx


'''Finalmente,podemos ver si están cerca u y u truncado'''


def comparison_u_utrunc(u_degree_np, u_degree_np_trunc):
    print("Estamos en comparison_u_utrunc")
    print("Are equal U_degree_np and U_degree_np_trunc:", np.allclose(u_degree_np, u_degree_np_trunc))


# *************************************** INTERPOLATION WITH SCIPY ************************************************


def interp_1d_vt(row_or_col, x, x_new, vt, interp):
    """

    :param row_or_col: Nos indica cómo se hará la interpolación (por filas o columnas)
                       Por fila, se va iterando de fila en fila y en cada fila se interpola un valor
                       (Valor calculado en esa fila para una columna nueva)
                       Por columna, se itera de columna en columna y para columna se calcula un valor nuevo de fila.
    :param x: Para cada fila/columna, su label corresponde a un valor de x.
    :param x_new: Label nuevo para calcular
    :param vt: Matriz que contiene las filas/columnas de datos.
    :param interp: Modo de interpolación, existen varias pero ahora mismo están interp1d y Akima.
    :return: Lista que contiene los valores del modo interpolado para x_new

    """
    print("Estamos en interp_1d_vt")
    n_range = vt.shape[0]  # Da igual 0/1 ya que la matriz vt será nxn o mxm
    list_interp_func_modes = []
    interpolated_mode = []

    for n in range(n_range):
        if row_or_col == "row":
            y = vt[n, :]
        elif row_or_col == "column":
            y = vt[:, n]
        if interp == "interp1d":
            list_interp_func_modes.append(intp.interp1d(x, y, kind='linear'))
        elif interp == "Akima":
            list_interp_func_modes.append(intp.Akima1DInterpolator(x, y))

        plt.plot(x, y)
        interpolated_mode.append(list_interp_func_modes[n](x_new))
        plt.scatter(x_new, interpolated_mode[n], s=100)
        plt.show()

    return interpolated_mode


# *************************************** RECONSTRUCTION VELOCITY FIELD **********************************************

# reconstrucción de las velocidades para el nuevo ángulo interpolado x_new


def reconstruction(interpolated_mode, u, smat):
    """
    :param interpolated_mode: Lista que contiene los valores numéricos del modo interpolado
    :param u: Vectores singulares izquierda
    :param smat: Singular values
    :return: Una lista que contiene los valores interpolados del field en str
    """

    print("Estamos en reconstruction")
    # Convertimos interp_modes a un array
    interp_modes = np.array(interpolated_mode)  # Convertimos interpolated_mode (es una lista) a un array
    new_field = np.dot(u, np.dot(smat, interp_modes))  # Calculamos el nuevo field
    new_field_list = new_field.tolist()  # Convertimos el array a lista
    comp_str = list(map(str, new_field_list))  # convertimos los valores numéricos a tipo str.

    return comp_str


def interp_lin(x, y, x_new):
    """
    function: interp_lin

    Descripción: Esta funcion realiza la interpolacion lienal entre dos vectores cuya correspondencia
                 depende del angulo de incidencia

    Input:

        x: Angulos de ataque
        y: Array de campo (p, T, vel, etc)
        x_new: Nuevo angulo de ataque.

    Output:
        y_new: Array del campo para el nuevo angulo de ataque x_new

    """

    x_new = np.cos(np.radians(float(x_new)))
    x1 = np.cos(np.radians(float(x[0])))
    x2 = np.cos(np.radians(float(x[1])))

    a = (x2 - x_new) / (x2 - x1)
    b = (x_new - x1) / (x2 - x1)

    y_new = a * y[:, 1] + b * y[:, 2]

    return y_new
