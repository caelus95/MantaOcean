#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 15:43:17 2021

@author: caelus
"""

PKG_path = '/home/caelus/dock_1/Working_hub/LGnDC_dep/python_cent/MantaPKG/'
import sys 
sys.path.append(PKG_path)
from Manta_Signals.procc_index import sig_pro, linearRegress4Cube
from Manta_Signals.utility import nc2npy
import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import dask

# =============================================================================
# Load data (Pararell)
# =============================================================================

r_path1 = '/home/caelus/dock_2/psi36/DATA/ncfile/GRSST/nc_files/'
# r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_GRSST/nc/'

nc_list = np.sort([file for file in os.listdir(r_path1) if file.endswith(".nc")])

data = xr.open_mfdataset(r_path1+'*.nc', parallel=True)

Sample_Data1 = xr.open_dataset(r_path1+nc_list[0])

data_Mm = data.resample(time="1MS").mean(dim="time")

del data

dask.config.set({"array.slicing.split_large_chunks": True})


Slice_lat, Slice_lon = [-10, 70], [112,260]

data_s_1 = data_Mm.loc[dict(lon=slice(Slice_lon[0],Slice_lon[1]),lat=slice(Slice_lat[0],Slice_lat[1]))]
data_s_2 = data_Mm.loc[dict(lon=slice(Slice_lon[0]-360,Slice_lon[1]-360),lat=slice(Slice_lat[0],Slice_lat[1]))]

data_s_2 = data_s_2.assign_coords(lon=( (data_s_2.lon + 360)  ))
data = xr.concat([data_s_1, data_s_2], dim="lon")

analysed_sst = data.analysed_sst.values
np.save(w_path1+'analysed_sst',analysed_sst_Mm)

uncertainty = data.analysis_uncertainty.values
mask = data.mask.values

'''
for i in tqdm(data.time.values):
    tmp_data = data.loc[dict(time=slice(i))].squeeze()
    tmp_name = str(i)[:7]

    tmp_data.to_netcdf(w_path1+'GRSST_Mm_'+tmp_name+'_10_70_112_260.nc')
'''

# =============================================================================
# 
# =============================================================================















# data_s = data.where( (data.lat>=minlat)&(data.lat<=maxlat)&(data.lon>=minlon)&(data.lon<=maxlon),
#                     drop=True)
# data_s = data.loc[dict(latitude=slice(maxlat,minlat),
#                        longitude=slice(minlon,maxlon),expver=1 )]


data_s = data_s.where(data_s.lsm==0,drop=False)

data_M = data_s.mean(dim='time')















































