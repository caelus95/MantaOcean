#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:41:50 2021

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

Output_path = '/home/caelus/dock_1/Working_hub/test_dep/room_1/His/' # his / avg ...

savedir = '/home/caelus/dock_1/Working_hub/test_dep/room_1/figs/'
time_starts = mdt.date2num(dt.datetime(1992,1,1))


list_dir = np.sort([file for file in os.listdir(Output_path) if file.endswith('.nc')])


plt.rcParams['font.size'] = 50
plt.rcParams['font.weight'] = 'bold'
plt.style.use('seaborn-darkgrid')
plt.rc('font',size=32)
plt.rcParams['axes.labelweight'] = 'bold'
for i in tqdm(list_dir):
    test_file = xr.open_dataset(Output_path+i,decode_times=False)
    # time = test_file['ocean_time']
    data = test_file['temp']
    sliced_data = data[:,-1,:,:]

    lon_rho,lat_rho = sliced_data.lon_rho.values, sliced_data.lat_rho.values

    T = sliced_data.ocean_time.values

    for j in T:
        time_convert = j/24/3600
        Model_time = mdt.num2date(time_starts+time_convert)
        data_ = sliced_data[sliced_data['ocean_time']==j].squeeze()
        
        plt.figure(figsize=(20,10))
        ax=plt.gca()
        m = Basemap(projection='cyl',llcrnrlat=lat_rho[0,0],urcrnrlat=lat_rho[-1,-1],
                        llcrnrlon=lon_rho[0,0],urcrnrlon=lon_rho[-1,-1],resolution='c',width=2)
        m.drawmapboundary(linewidth=3)
        x, y = m(lon_rho, lat_rho)
        m.fillcontinents(color='black',lake_color='black')
        m.drawcoastlines()
        m.drawparallels(np.arange(lat_rho[0,0],lat_rho[-1,-1],10),labels=[True,False,False,False],
                        dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
        m.drawmeridians(np.arange(round(lon_rho[0,0]+5,-1),lon_rho[-1,-1],20.),labels=[False,False,False,True],
                        dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
        title_time = mdt.num2date(time_starts+time_convert).strftime('%Y-%m-%d %M:%S')
        plt.title(title_time[:16], position=(0.5, 1.0+0.05),fontweight='bold',fontsize=46)
        
        cs = m.pcolormesh(x,y,data_,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
        plt.clim(0,32)
        ax.tick_params(labelcolor='k',labelsize=22,pad=15,grid_linewidth=2)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.1)
        cax.tick_params(labelsize=18,width='bold')
        cax.set_ylabel('',{'fontsize':32,'fontweight':'bold','style':'italic'})
        #label 
        h = plt.colorbar(label='$\degree$C',cax=cax);
        plt.savefig(savedir+title_time[:10],dpi=200)
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

        
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
