# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:51:14 2020

@author: psi36
"""



'''
Wind ERA5
'''

import os
import numpy as np
import time
import datetime as dt
import pandas as pd
from netCDF4 import Dataset
from scipy import io
from numpy import hstack



Directory = 'F:/psi36/DATA/ncfile/'
nc_file = 'ER5_winduv.nc'

Data = Dataset(Directory+nc_file)


llcrnrlat=0
urcrnrlat=32
llcrnrlon=112
urcrnrlon=150
t_len = 30

lat = Data['latitude'][:]
lon = Data['longitude'][:]

lat_co=hstack(np.where((lat>=llcrnrlat)&(lat<=urcrnrlat))[0])
lon_co=hstack(np.where((lon>=llcrnrlon)&(lon<=urcrnrlon))[0])

lat_rgnl=lat[lat_co]
lon_rgnl=lon[lon_co]

u10 = Data.variables['u10'][:,lat_co,lon_co]
v10 = Data.variables['v10'][:,lat_co,lon_co]

u10[u10<-1000] = np.nan
v10[v10<-1000] = np.nan


WesternBound_ERA5_lonlat={'lat':lat,'lon':lon,'lat_co':lat_co,'lon_co':lon_co}


io.savemat('F:/psi36/DATA/Kuroshio_NEC/WesternBound_ERA5_u10.mat', {'u10':u10})
io.savemat('F:/psi36/DATA/Kuroshio_NEC/WesternBound_ERA5_v10.mat', {'v10':v10})

io.savemat('F:/psi36/DATA/Kuroshio_NEC/WesternBound_ERA5_lonlat.mat', WesternBound_ERA5_lonlat)

plt.quiver(u10[0,:,:],v10[0,:,:])
plt.contourf(u10[0,:,:])










