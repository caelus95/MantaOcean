#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:26:49 2021

@author: caelus
"""

import numpy as np
import xarray as xr
from tqdm import tqdm
import os 
import shutil
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
llcrnrlat = 0 # int  
urcrnrlat = 60 # int
llcrnrlon1 = 112 # int
urcrnrlon1 = 180 # inta
llcrnrlon2 = -180 # int
urcrnrlon2 = -80 
var_name1 = 'analysed_sst'
var_name3 = 'analysis_uncertainty'
var_name2 = 'mask'

r_path1 = '/home/caelus/dock_2/psi36/DATA/ncfile/GRSST/nc_files/Monthly/'
w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/'

nc_list = np.sort([file for file in os.listdir(r_path1) if file.endswith(".nc")])

Sample_data = Dataset(r_path1+nc_list[0])

ELat = Sample_data.variables['lat'][:]
ELon = Sample_data.variables['lon'][:]

lat_co = np.where((ELat >= llcrnrlat) & (ELat <= urcrnrlat))[0]
lon_co1 = np.where((ELon >= llcrnrlon1) & (ELon <= urcrnrlon1))[0]
lon_co2 = np.where((ELon >= llcrnrlon2) & (ELon <= urcrnrlon2))[0]

lat_rgnl = ELat[lat_co]
lon_rgnl1 = ELon[lon_co1]
lon_rgnl2 = ELon[lon_co2]


Data1_1,n = np.zeros([len(nc_list),len(lat_co), len(lon_co1)]),0
Data1_2 = np.zeros([len(nc_list),len(lat_co), len(lon_co2)])
Data3_1,Data3_2 = np.zeros_like(Data1_1), np.zeros_like(Data1_2) 
for i in tqdm(nc_list):
    tmp_data = Dataset(r_path1+i)
    Data1_1[n, :, :] = tmp_data.variables[var_name1][0, lat_co, lon_co1]#.data[0]
    Data1_2[n, :, :] = tmp_data.variables[var_name1][0, lat_co, lon_co2]#.data[0]
    Data3_1[n, :, :] = tmp_data.variables[var_name3][0, lat_co, lon_co1]#.data[0]
    Data3_2[n, :, :] = tmp_data.variables[var_name3][0, lat_co, lon_co2]#.data[0]
    n+=1

Data2_1 = tmp_data.variables[var_name2][0, lat_co, lon_co1]
Data2_2 = tmp_data.variables[var_name2][0, lat_co, lon_co2]
data2_asia_centered = np.concatenate([Data2_1,Data2_2],axis=1)

data1_asia_centered = np.concatenate([Data1_1,Data1_2],axis=2)
data3_asia_centered = np.concatenate([Data3_1,Data3_2],axis=2)

data3_mean = np.nanmean(data3_asia_centered,axis=0)

lon_rgnl = np.concatenate([lon_rgnl1,lon_rgnl2+360],axis=0)

ds = xr.Dataset(
    {
        'sst': (["time","lat", "lon"], data1_asia_centered),
        
        "mask": (["lat","lon"],data2_asia_centered.data),
        "analysis_uncertainty": (["time","lat", "lon"], data3_asia_centered)
    },
    coords={
        "lon": lon_rgnl,
        "lat": lat_rgnl,
        "time": pd.date_range("1993-01-01", periods=312),
        # "reference_time": pd.Timestamp("2014-09-05"),
    },
)

ds.to_netcdf(path=w_path1+'GRSST.nc',mode='w')

os.mkdir(w_path1+'GHRSST_199301_201812')
np.save(w_path1+'GHRSST_199301_201812/analysed_sst_0_60_112_280',data1_asia_centered)
np.save(w_path1+'GHRSST_199301_201812/analysis_uncertainty',data3_asia_centered)
np.save(w_path1+'GHRSST_199301_201812/mask',data2_asia_centered.data)
np.save(w_path1+'GHRSST_199301_201812/lon',lon_rgnl.data)
np.save(w_path1+'GHRSST_199301_201812/lat',lat_rgnl.data)














