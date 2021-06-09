#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:41:45 2021

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



r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/' 


Data = xr.open_dataset(r_path1+'adt__20_65_112_280_M.nc')



Time = ['2006-07','2008-10'] 
# minlon,maxlon = 110,280
# minlat,maxlat = -60,60,


data_s = Data.loc[dict(time=slice(Time[0],Time[1]))]
data_s = data_s.assign_coords(time=np.arange(len(data_s.time.values)))



# define a function to compute a linear trend of a timeseries
def linear_trend(x):
    pf = np.polyfit(x.time, x, 1)
    # need to return an xr.DataArray for groupby
    return xr.DataArray(pf[0])

# stack lat and lon into a single dimension called allpoints
stacked = data_s.adt.stack(allpoints=['lat','lon'])
# apply the function over allpoints to calculate the trend at each point
trend = stacked.groupby('allpoints').apply(linear_trend)
# unstack back to lat lon coordinates
trend_unstacked = trend.unstack('allpoints')




plt.figure()
plt.pcolormesh(trend_unstacked,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-.01,.01)
plt.title('Trend 2006-07 ~ 2008-10')
plt.show()



































