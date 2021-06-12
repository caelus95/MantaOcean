#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:20:38 2021

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



# define a function to compute a linear trend of a timeseries
def linear_trend(x):
    pf = np.polyfit(x.time, x, 1)
    # need to return an xr.DataArray for groupby
    return xr.DataArray(pf[0])


data = xr.open_dataset('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/adt_0_60_112_260_M.nc',decode_times=True)

Time = ['1993-01','2019-12'] 

data_s = data.loc[dict(time=slice(Time[0],Time[1]))]

data_s = data_s.assign_coords(time=np.arange(len(data_s.time.values)))


lon,lat = data.adt.lon, data.adt.lat


# stack lat and lon into a single dimension called allpoints
stacked = data_s.adt.stack(allpoints=['y','x'])
# apply the function over allpoints to calculate the trend at each point
trend = stacked.groupby('allpoints').apply(linear_trend)
# unstack back to lat lon coordinates
trend_unstacked = trend.unstack('allpoints')



plt.figure(figsize=(26, 5))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],\
            llcrnrlon=lon[0],urcrnrlon=lon[-1],resolution='i')
lon2, lat2 = np.meshgrid(lon,lat)
x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(5.,33.,5.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=18,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=18,fontweight='bold',color='grey')
plt.title('Trend'+str(Time[0])+'~'+str(Time[1]), fontproperties='', position=(0.5, 1.0+0.07), fontsize=34,fontweight='bold')
# plt.suptitle(' UV & speed ',fontstyle='italic',position=(0.5, .92),fontsize=20)
cs = m.pcolormesh(x,y,trend_unstacked,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-.001,.001)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
# plt.savefig('F:/psi36/DATA/temp_var3/meanflow',bbox_inches='tight')
plt.show()






















