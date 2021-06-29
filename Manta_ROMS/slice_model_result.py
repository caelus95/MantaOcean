#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 07:12:35 2021

@author: caelus
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 05:43:33 2021

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
import warnings


r_path1 = '/home/caelus/dock_1/Working_hub/ROMS_dep/ROMS/'

Data1 = xr.open_dataset(r_path1+'u_eastward_Mm.nc')
Data2 = xr.open_dataset(r_path1+'v_northward_Mm.nc')

Time = ['1993-01','2019-12'] 
minlon,maxlon = 125,260
minlat,maxlat = 10,30


lon = Data1.lon_rho.values[0,:]
lat = Data1.lat_rho.values[:,0]


lat_co = np.where((lat >= minlat) & (lat <= maxlat))[0]
lon_co = np.where((lon >= minlon) & (lon <= maxlon))[0]

u_s = Data1.loc[dict(ocean_time=slice(Time[0],Time[1]),eta_rho=lat_co,xi_rho=lon_co)]
v_s = Data2.loc[dict(ocean_time=slice(Time[0],Time[1]),eta_rho=lat_co,xi_rho=lon_co)]


speed = (u_s.u_eastward**2 + v_s.v_northward**2)/2

S = speed.values
np.save(r_path1+'EKE',S)
np.save(r_path1+'lon_EKE',lon[lon_co])
np.save(r_path1+'lat_EKE',lat[lat_co])

# -----------------

# =============================================================================
# 
# =============================================================================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
















