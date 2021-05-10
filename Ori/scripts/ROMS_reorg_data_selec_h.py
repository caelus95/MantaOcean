# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:00:58 2021

@author: shamu

Usage : Horizontally sellect area 

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

savedir = '/home/caelus/dock_1/Working_hub/test_dep/room_1/figdsas/'

minlon,maxlon = 112,180
minlat,maxlat = -10,20
data_name = 'temp'
factor = [6,6] # Reduce vectors
vectors = 1
season = 0
time_coords = False
coords_path = '' # ~.npy

data1 = xr.open_dataset(r_path1)
data2 = xr.open_dataset(r_path2)
data3 = xr.open_dataset(r_path3)


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


data = xr.merge([data1,data2,data3]) 


data_s = data.where( (data.lat>=minlat)&(data.lat<=maxlat)&(data.lon>=minlon)&(data.lon<=maxlon),
                    drop=True)

data_1d = data_s[data_name].mean(axis=2).mean(axis=1)

mean_2d = data_s.mean(axis=0)

data_a = data_s - mean_2d

data_ = data_a


if season:
    data_ = data_.groupby('time.season').mean()
    t = data_.season
    time_name = 'season'

elif time_coords:
    tmp_coords = np.load(coords_path)
else: 
    t = data_.time
    time_name = 'time'


# =============================================================================
# 1d figure
# =============================================================================
#plt.params
plt.figure()
plt.plot(t,data_1d)
# plt.figsave()


for i in t:
    tmp_data = data_.where(data_[time_name]==i,drop=True).squeeze()
    if season:
        t_str = str(i.values)
    else:
        t_str = np.datetime_as_string(i)[:7]
    plt.figure(figsize=(20,13))
    ax=plt.gca()
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
    cb = m.contour(x,y,tmp_data['temp'].values,colors='k',linestyles='-.',level=5)
    plt.clabel(cb,inline=1,fontsize=24,fmt='%d')
    ca = m.pcolormesh(x,y,tmp_data['temp'],cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    if vectors:
        lat_,lon_,u_,v_= r_vectors(tmp_data.lat,tmp_data.lon,tmp_data.u_eastward,tmp_data.v_northward,factor)
        x_,y_= m(lon_,lat_)
        m.quiver(x_,y_,u_,v_,zorder=1,scale=15,width=0.003,headlength=6)
    plt.clim(0,32)
    ax.tick_params(labelcolor='k',labelsize=22,pad=15,grid_linewidth=2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=18)
    cax.set_ylabel(ca,{'fontsize':32,'fontweight':'bold','style':'italic'})
    h = plt.colorbar(label='$\degree$C',cax=cax)
    

    #label 
    
    # plt.savefig(savedir+title_time[:10],dpi=200)
    















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

        
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
