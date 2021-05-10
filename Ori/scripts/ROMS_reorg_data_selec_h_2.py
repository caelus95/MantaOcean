#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:25:04 2021

@author: caelus
"""



import numpy as np
import pandas as pd
import xarray as xr
import os 
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.dates as mdt
import datetime as dt
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')

r_path1 = '/home/caelus/dock_1/Working_hub/ROMS_dep/task_2/data/tempMm.nc' # his / avg ...
r_path2 = '/home/caelus/dock_1/Working_hub/ROMS_dep/task_2/data/u_eastwardMm.nc'
r_path3 = '/home/caelus/dock_1/Working_hub/ROMS_dep/task_2/data/v_northwardMm.nc'
r_path4 = '/home/caelus/dock_1/Working_hub/ROMS_dep/task_2/data/zetaMm.nc'

savedir = '/home/caelus/wormhole/figs2/'

minlon,maxlon = 112,140
minlat,maxlat = 8,45
data_name_1 = 'temp'
data_name_2 = 'zeta'
factor = [3,3] # Reduce vectors
vectors = 1
season = 1
All_Time_Mean = 1
# time_coords = False
coords_path = '' # ~.npy

data1 = xr.open_dataset(r_path1)
data2 = xr.open_dataset(r_path2)
data3 = xr.open_dataset(r_path3)
data4 = xr.open_dataset(r_path4)


def r_vectors(x,y,data1,data2,factor):
    import numpy as np
    xx = x.values.shape[0]
    yy = y.values.shape[0]
    a = np.arange(0,xx,factor[0])
    b = np.arange(0,yy,factor[1])
    return x[a], y[b], data1[a,b], data2[a,b]

# lat_,lon_,u_,v_= r_vectors(tmp_data.lat,tmp_data.lon,tmp_data.u_eastward,tmp_data.v_northward,factor)

# # =============================================================================

# x = tmp_data.lat
# y = tmp_data.lon
# data1 = tmp_data.u_eastward
# data2 = tmp_data.v_northward
# ============================================================================


data = xr.merge([data1,data2,data3,data4]) 


data_s = data.where( (data.lat>=minlat)&(data.lat<=maxlat)&(data.lon>=minlon)&(data.lon<=maxlon),
                    drop=True)

data_1d = data_s[data_name_2].mean(dim='lat').mean(dim='lon')

mean_2d = data_s.mean(dim='time')

data_a = data_s - mean_2d

data_ = data_s


if season:
    data_ = data_.groupby('time.season').mean()
    t = data_.season
    time_name = 'season'

# elif time_coords:
#     tmp_coords = np.load(coords_path)
else: 
    t = data_.time
    time_name = 'time'

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# =============================================================================
# 1d figure
# =============================================================================
#plt.params
if season:
    pass
else:
    plt.figure()
    plt.plot(t,data_1d)
    # plt.figsave()


for i in t:
    tmp_data = data_.where(data_[time_name]==i,drop=True).squeeze()
    if season:
        t_str = str(i.values)
    else:
        t_str = np.datetime_as_string(i)[:7]
    # plt.figure(figsize=(20,13))
    # ax=plt.gca()
    fig, ax = plt.subplots(figsize=(20,15),linewidth=5)
    m = Basemap(projection='cyl',llcrnrlat=tmp_data.lat[0],urcrnrlat=tmp_data.lat[-1],
                    llcrnrlon=tmp_data.lon[0],urcrnrlon=tmp_data.lon[-1],resolution='c',width=2)
    m.drawmapboundary(linewidth=3)
    x, y = np.meshgrid(tmp_data.lon, tmp_data.lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(int(tmp_data.lat[0].values)+.5,tmp_data.lat[-1].values,5),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(int(tmp_data.lon[0].values)-.5,tmp_data.lon[-1].values,10),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
    plt.title(t_str, position=(0.5, 1.0+0.05),fontweight='bold',fontsize=46)
    cb = m.contour(x,y,tmp_data[data_name_1].values,colors='k',linestyles='-.',level=5)
    plt.clabel(cb,inline=1,fontsize=24,fmt='%d')
    ca = m.pcolormesh(x,y,tmp_data[data_name_1],cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    if vectors:
        lat_,lon_,u_,v_= r_vectors(tmp_data.lat,tmp_data.lon,tmp_data.u_eastward,tmp_data.v_northward,factor)
        x_,y_= m(lon_,lat_)
        cc = m.quiver(x_,y_,u_,v_,zorder=1,scale=10,width=0.003,headlength=5.5,headwidth=5)
        ax.quiverkey(cc,X=.05,Y=1.045, U = .5 ,label='', labelpos='E')
        ax.text(115.8,20.8,'50 cm/s',fontsize=20,fontweight='bold')
    
    plt.clim(5,33.)
    ax.tick_params(labelcolor='k',labelsize=22,pad=15,grid_linewidth=2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=18)
    cax.set_ylabel(ca,{'fontsize':32,'fontweight':'bold','style':'italic'})
    h = plt.colorbar(label='$\degree$C',cax=cax)
    plt.savefig(savedir+t_str,dpi=200)
    plt.show()

    #label 
    
    # plt.savefig(savedir+title_time[:10],dpi=200)
    

# =============================================================================
# MEAN PLot
# =============================================================================


if All_Time_Mean:
    tmp_data = mean_2d
    
    fig, ax = plt.subplots(figsize=(20,15),linewidth=5)
    m = Basemap(projection='cyl',llcrnrlat=tmp_data.lat[0],urcrnrlat=tmp_data.lat[-1],
                     llcrnrlon=tmp_data.lon[0],urcrnrlon=tmp_data.lon[-1],resolution='c',width=2)
    m.drawmapboundary(linewidth=3)
    x, y = np.meshgrid(tmp_data.lon, tmp_data.lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(int(tmp_data.lat[0].values)+.5,tmp_data.lat[-1].values,5),labels=[True,False,False,False],
                     dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(int(tmp_data.lon[0].values)-.5,tmp_data.lon[-1].values,10),labels=[False,False,False,True],
                     dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
    plt.title('All Time mean', position=(0.5, 1.0+0.05),fontweight='bold',fontsize=46)
    cb = m.contour(x,y,tmp_data[data_name_1].values,colors='k',linestyles='-.',level=5)
    plt.clabel(cb,inline=1,fontsize=24,fmt='%d')
    ca = m.pcolormesh(x,y,mean_2d[data_name_1],cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    if vectors:
         lat_,lon_,u_,v_= r_vectors(tmp_data.lat,mean_2d.lon,tmp_data.u_eastward,tmp_data.v_northward,factor)
         x_,y_= m(lon_,lat_)
         cc = m.quiver(x_,y_,u_,v_,zorder=1,scale=10,width=0.003,headlength=5.5,headwidth=5)
         ax.quiverkey(cc,X=.05,Y=1.045, U = .5 ,label='', labelpos='E')
         ax.text(115.8,20.8,'50 cm/s',fontsize=20,fontweight='bold')
    plt.clim(20,33.)
    ax.tick_params(labelcolor='k',labelsize=22,pad=15,grid_linewidth=2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=18)
    cax.set_ylabel(ca,{'fontsize':32,'fontweight':'bold','style':'italic'})
    h = plt.colorbar(label='$\degree$C',cax=cax)
    plt.savefig(savedir+'Total_mean',dpi=200)
    plt.show()

# =============================================================================
# 
# =============================================================================





#    plt.show()

'''

#for i in list_dir:
test_file = xr.open_dataset(Output_path+i,decode_times=False)
time = test_file['ocean_time']
temp = test_file['temp']
slice_temp = temp[:,-1,:,:]

lon_rho,lat_rho = slice_temp.lon_rho.values, slice_temp.lat_rho.values

len(slice_temp.ocean_time.values)
T = slice_temp.ocean_time.values
t,at,on = slice_temp.values.shape




print('Hellow world')
import matplotlib.dates as mdt
import datetime as dt

time_starts = mdt.date2num(dt.datetime(1992,1,1))

print('=======================')
for i in T:
    time_convert = i/24/3600
    print(mdt.num2date(time_starts+time_convert))
'''

        
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
