#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 22:19:23 2021

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

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/ERA5/' 


Time = ['1993-01','2019-12'] 
minlon,maxlon = 112,260
minlat,maxlat = -10,70,
    

Sample_Data1 = xr.open_dataset(r_path1+'ERA5_landSea_mask.nc')
Sample_Data2 = xr.open_dataset(r_path1+'ERA5_single_level.nc')

data = xr.merge([Sample_Data1,Sample_Data2])


data_s = data.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(maxlat,minlat),
                        longitude=slice(minlon,maxlon),expver=1 )]

data_s = data_s.where(data_s.lsm==0,drop=False)

# data_M = data_s.mean(dim='time')


curlZ_s = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/tmp/curlZ_s.npy') 


CURL = xr.Dataset(
    {
        'curlZ': (["time","latitude", "longitude"], curlZ_s)#,
        # "mask": (["y","x"],mask)
    },
    coords={
        "longitude": (["longitude"], data_s.longitude),
        "latitude": (["latitude"], data_s.latitude),
        "time": (['time'], data_s.time),
        # "reference_time": pd.Timestamp("2014-09-05"),
    },
)



Taiwan_pole = CURL.loc[dict(latitude=slice(27,22),longitude=slice(155,180))].mean(dim=['latitude','longitude'])
philippines_pole = CURL.loc[dict(latitude=slice(13,8),longitude=slice(130,155))].mean(dim=['latitude','longitude'])

PTO_index= philippines_pole -Taiwan_pole

PTO_index
PTO_index_2Y = PTO_index.rolling(time=24,center=True).mean().dropna("time")

PTO_index = PTO_index.curlZ.values

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/PTO_index',PTO_index)
















