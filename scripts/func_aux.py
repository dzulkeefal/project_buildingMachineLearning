# -*- coding: utf-8 -*-

"""
Created on Tue Mar  7 10:26:56 2023

@author: Workstation5
"""
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def cumulative(s):
    cu_list = []
    length = len(s)
    cu_list = [sum(s[0:x:1]) for x in range(0, length + 1)]
    return cu_list[1:]


def vtk_extract_field_points(dir_file):
    # Abrimos el actual fichero .vtk
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(dir_file)
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

    return points_np, scalarField_vector
