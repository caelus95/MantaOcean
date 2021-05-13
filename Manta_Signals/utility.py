#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:25:44 2021

@author: caelus
"""



def nc2npy(file_name,Mask=False):
    '''
    Inputs :
        file_name --> data_path+data_name
    '''
    import xarray as xr
    import numpy as np
    dataset = xr.open_dataset(file_name)
    print(dataset)
    data_list,coor_list = [],[]
    for i in list(dataset):
        data_list.append(dataset[i].values)
    for j in list(dataset.coords):
        print(j)
        coor_list.append(dataset[j].values)
    n=8
    print('='*n+' Variables list '+'='*n)
    print(list(dataset))
    print('='*n+' Coordinates list '+'='*n)
    print(list(dataset.coords))
    print('='*n+' Outputs order '+'='*n)
    print(list(dataset)+list(dataset.coords))
    return data_list,coor_list
    