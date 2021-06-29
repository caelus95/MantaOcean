#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 14:37:28 2021

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
from scipy import signal
import xscale

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/CDS/'
w_path1 = r_path1

ADT_t = xr.open_dataset(r_path1+'T_CDS_monthly_199301_201912.nc',decode_times=False)
ADT_t = ADT_t.drop(['crs','lat_bnds','lon_bnds','err','sla','ugosa','vgosa'])

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)

    return da - fit

def detrend(da, dims, deg=1):
    # detrend along multiple dimensions
    # only valid for linear detrending (deg=1)
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

DATA_list = ['adt','ugos','vgos']

tmp_DATASET = []
for i in tqdm(DATA_list):
    print(i)
    tmp_data = detrend_dim(ADT_t[i],'time')
    tmp_DATASET.append(tmp_data.to_dataset(name=i))

DATASET = xr.merge([tmp_DATASET[0],tmp_DATASET[1],tmp_DATASET[2]])

DATASET.to_netcdf(path=w_path1+'Detrended_CDS_monthly_199301_201912.nc',mode='w')




# --------------- TEST --------------------------------
# for i in range(720):
#     for j in range(1400):
#         print(i,j)
#         detrend_dim(ADT_t.adt[:,i,j],dim='time')
        
# A = ADT_t.adt[:,500,800].values
# B = detrend_dim(ADT_t.adt[:,500,800],dim='time').values
# A_m = A.mean()

# plt.plot(A)
# plt.plot(B)
# plt.axhline(y=0,color='k',linestyle='-.')
# plt.axhline(y=A_m,color='k',linestyle='-.')

