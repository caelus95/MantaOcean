#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:33:07 2021

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

# =============================================================================
# WSC & Press
# =============================================================================

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


data_WSC = xr.merge([data_s,CURL])

data_WSC = data_WSC.drop(['sp','tp','u10','v10','lsm'])


# =============================================================================
# 
# =============================================================================
# mask for UV
def mask4xarray(lon_list,lat_list,data):
    from mpl_toolkits.basemap import Basemap
    fig, ax = plt.subplots(linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=70,\
                llcrnrlon=112,urcrnrlon=280,resolution='c')
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=8,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=8,fontweight='bold',color='grey')
    m.plot([lon_list[0],lon_list[1],lon_list[1],lon_list[0],lon_list[0]],
           [lat_list[0],lat_list[0],lat_list[1],lat_list[1],lat_list[0]],
           color='r',linestyle='--',linewidth=3)
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=lat_list[0]-10,urcrnrlat=lat_list[1]+10,\
                llcrnrlon=lon_list[0]-10,urcrnrlon=lon_list[1]+10,resolution='c')
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=8,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,5.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=8,fontweight='bold',color='grey')
    m.plot([lon_list[0],lon_list[1],lon_list[1],lon_list[0],lon_list[0]],
           [lat_list[0],lat_list[0],lat_list[1],lat_list[1],lat_list[0]],
           color='r',linestyle='--',linewidth=3)
    plt.tight_layout()
    plt.show()

    tmp_var = input('........? : ')
    if tmp_var == 'y':
        pass
    else :
        raise 
    #------------------------
    data_s = data.where( (data.longitude<lon_list[0]) | (data.longitude>lon_list[1]) |\
              (data.latitude<lat_list[0]) | (data.latitude>lat_list[1]),drop=False)
    #------------------------
    tmp = data_s.fillna(9999)
    tmp = tmp.mean(dim='time')
    
    fig, ax = plt.subplots(linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=70,\
                llcrnrlon=112,urcrnrlon=280,resolution='c')
    # x, y = m(lon, lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=8,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=8,fontweight='bold',color='grey')
    m.pcolormesh(tmp.longitude.values,tmp.latitude.values,tmp.curlZ.values)
    plt.tight_layout()
    plt.show()
    return data_s

Masked_data = mask4xarray([112,280],[-10,10],data_WSC)
Masked_data = mask4xarray([112,120],[10,45],Masked_data)
Masked_data = mask4xarray([112,130],[30,45],Masked_data)
Masked_data = mask4xarray([112,280],[32,70],Masked_data)

Masked_data1 = mask4xarray([196,208],[14,24],Masked_data)


### Drop nan
Masked_data1 = Masked_data1.loc[dict(latitude=slice(35,10),longitude=slice(120,260))]

Masked_data_a = Masked_data1 - Masked_data1.mean(dim='time')

Sample_figdata = Masked_data1.mean(dim='time')
Sample_figdata = Masked_data_a.curlZ[0,:,:]
# Sample figure
fig, ax = plt.subplots(figsize=(16,8.5),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
cs2 = m.pcolormesh(Sample_figdata.longitude.values,np.flipud(Sample_figdata.latitude.values),
                   np.flipud(Sample_figdata.values)*10**7,shading='gouraud')
# plt.clim(np.nanmin(Sample_figdata.values),np.nanmax(Sample_figdata.values))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
h = plt.colorbar(label='$\mathit{10^{-7} N \cdot m^{-3}}$',cax=cax);
plt.tight_layout()
plt.show()



































