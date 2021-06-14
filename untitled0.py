#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:26:28 2021

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




data_WSC = xr.merge([data_s,CURL])

data_WSC = data_WSC.drop(['sp','tp','lsm'])

data_WSC_a = data_WSC - data_WSC.mean(dim='time')

data_WSC_a = data_WSC_a.fillna(-999)

WD = 2*12
data_WSC_a_2Y = data_WSC_a.rolling(time=WD,center=True).mean().dropna("time")

data_WSC_a_2Y = data_WSC_a_2Y.where(data_WSC_a_2Y!=-999,drop=False) 



# Def r_vector


x,y = data_WSC_a_2Y.longitude, data_WSC_a_2Y.latitude
data1, data2 = data_WSC_a_2Y.u10, data_WSC_a_2Y.v10
factor = [6, 6]

def r_vector4cube(x,y,data1,data2,factor):
    xx = x.values.shape[0]
    yy = y.values.shape[0]
    a = np.arange(0,xx,factor[0])
    b = np.arange(0,yy,factor[1])
    
    r_x, r_y = x[a], y[b]
    
    r_data1 = data1.where( (data1.longitude==r_x) & (data1.latitude==r_y), drop=True )
    r_data2 = data2.where( (data2.longitude==r_x) & (data2.latitude==r_y), drop=True )    
    
    return r_x, r_y, r_data1, r_data2





figdata11 = data_WSC_a_2Y.msl.values/100 # Pa --> hpa
figdata12 = data_WSC_a_2Y.curlZ.values
lon_10 = data_WSC_a_2Y.longitude.values  
lat_10 = np.flipud(data_WSC_a_2Y.latitude.values)
lon_m10, lat_m10 = np.meshgrid(lon_10,lat_10)

# ------------ test fig WSC ----------------

fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# lon_m, lat_m = np.meshgrid(lon_00,lat_00)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
# plt.title(' Climatological WSC & Pressure ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m10,lat_m10,np.flipud(figdata11[70,:,:]),colors='grey',linewidths=2.5,levels=10)
plt.clim(-3.3,3.3)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(figdata12[70,:,:])*10**7,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)
# cs2 = m.pcolormesh(lon_m,lat_m,mapdata_2,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(min_mapdata_2,max_mapdata_2)
# m.quiver(lon_mr,lat_mr,data_mur,data_mvr,headwidth=5,headaxislength=10,headlength=10)
#m.quiver(Acp,Bcp,u_s,v_s,scale=7,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
h = plt.colorbar(label='$\mathit{10^{-7} N \cdot m^{-3}}$',cax=cax);
# plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
plt.tight_layout()
plt.show()


