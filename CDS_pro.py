#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 18:03:50 2021

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

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/CDS/'
data_name = 'Detrended_CDS_monthly_199301_201912.nc'
w_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Detrend/data/'
w_name = 'Detrended_CDS_NP_Total.nc'

minlon,maxlon = 112,280
minlat,maxlat = -10,70

# data_a_6M = data_a_6M.mean(dim='latitude')
def MantaCurl2D(u,v,dx=28400.0,dy=28400.0 ):
    import numpy as np
    '''
    dx = 28400.0 # meters calculated from the 0.25 degree spatial gridding 
    dy = 28400.0 # meters calculated from the 0.25 degree spatial gridding 
    '''
    u_T = u.transpose([1,0])
    v_T = v.transpose([1,0])

    
    du_dx, du_dy = np.gradient(u_T, dx,dy)
    dv_dx, dv_dy = np.gradient(v_T, dx,dy)

    curl = dv_dx - du_dy
    return curl.transpose([1,0])


ADT_t = xr.open_dataset(r_path+data_name,decode_times=True)

ADT_t = ADT_t.loc[dict(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon))]

# Calculating Vorticity (Curl) 

tmp_ugos = ADT_t.ugos.values
tmp_vgos = ADT_t.vgos.values

t,at,on = tmp_ugos.shape
Curl = np.zeros_like(tmp_ugos)
for i in range(t):
    Curl[i,:,:] = MantaCurl2D(tmp_ugos[i,:,:],tmp_vgos[i,:,:])

CURL = xr.Dataset(
    {
        'curl': (["time","latitude", "longitude"], Curl)#,
        # "mask": (["y","x"],mask)
    },
    coords={
        "longitude": (["longitude"], ADT_t.longitude),
        "latitude": (["latitude"], ADT_t.latitude),
        "time": (['time'], ADT_t.time),
        # "reference_time": pd.Timestamp("2014-09-05"),
    },)


# Calculating EKE
ADT_t['EKE'] = (ADT_t.ugos*2 + ADT_t.vgos*2)/2

# Merge data
ADT_t = xr.merge([ADT_t,CURL])


ADT_t.to_netcdf(w_path+w_name,'w')






























