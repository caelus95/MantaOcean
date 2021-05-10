#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:35:41 2021

@author: caelus
"""

fig, ax = plt.subplots(figsize=(7.5,6.5 ),linewidth=5)
ax.plot(map_data[:,0],map_data[:,1],color='k')

Q = ax.quiver(Lon,Lat, Uwind_kuri_S1C1 - Uwind_kuri_S1Cr, Vwind_kuri_S1C1 - Vwind_kuri_S1Cr,units='xy',
              scale=5,color='r',width=0.10,zorder = 1)    # width=0.003, scale=5000,

Q1 = ax.quiver(Lon[0,0],Lat[0,0],Uwind[0,0],Vwind[0,0],units='xy',scale=5,
               color='r',width=0.15,zorder = 1)

ax.quiverkey(Q1, X = 0.8, Y = -0.1,  U = 10, label='', labelpos="E")  
ax.text(139.5,16.5,'10 knot',fontsize=20,fontweight='bold')

plt.yticks(np.arange(10,51,5))
ax.set_yticklabels(np.arange(10,51,5),fontweight='bold',fontsize=13)
plt.xticks(np.arange(110,161,5))
ax.set_xticklabels(np.arange(110,161,5),fontweight='bold',fontsize=13)




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


tmp_data = data_s


def r_vectors(x,y,data1,data2,factor):
    import numpy as np
    xx = x.values.shape[0]
    yy = y.values.shape[0]
    a = np.arange(0,xx,factor[0])
    b = np.arange(0,yy,factor[1])
    return x[a], y[b], data1[a,b], data2[a,b]

# lat_,lon_,u_,v_= r_vectors(tmp_data.lat,tmp_data.lon,tmp_data.u_eastward,tmp_data.v_northward,factor)

# x= tmp_data.lat
# y= tmp_data.lon
# data1 = tmp_data.u_eastward
# data2 = tmp_data.v_northward

UU = tmp_data.u_eastward.mean(axis=0)
VV= tmp_data.v_northward.mean(axis=0)
lat_,lon_,u_,v_= r_vectors(tmp_data.lat,tmp_data.lon,UU,VV,[6,6])

# plt.figure(figsize=(20,13))


fig, ax = plt.subplots(figsize=(20,15),linewidth=5)

# ax=plt.gca()
m = Basemap(projection='cyl',llcrnrlat=tmp_data.lat[0],urcrnrlat=tmp_data.lat[-1],
                llcrnrlon=tmp_data.lon[0],urcrnrlon=tmp_data.lon[-1],resolution='c',width=2)
m.drawmapboundary(linewidth=3)
x, y = np.meshgrid(tmp_data.lon, tmp_data.lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(int(tmp_data.lat[0].values)+.5,tmp_data.lat[-1].values,5),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(int(tmp_data.lon[0].values)-.5,tmp_data.lon[-1].values,15),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
plt.title('AHhhhhhhhhhhh', position=(0.5, 1.0+0.05),fontweight='bold',fontsize=46)
cb = m.contour(x,y,tmp_data['temp'].values[0,:,:],colors='k',linestyles='-.',level=5)
plt.clabel(cb,inline=1,fontsize=24,fmt='%d')
ca = m.pcolormesh(x,y,tmp_data['temp'][0,:,:],cmap=plt.cm.get_cmap('jet'),shading='gouraud')
# lat_,lon_,u_,v_= r_vectors(tmp_data.lat,tmp_data.lon,tmp_data.u_eastward,tmp_data.v_northward,factor)
x_,y_= m(lon_,lat_)
AAA = m.quiver(x_,y_,u_,v_,zorder=1,scale=10,width=0.003,headlength=5.5,headwidth=5)
ax.quiverkey(AAA,X=.05,Y=1.045, U = .5 ,label='', labelpos='E')
ax.text(115.8,20.8,'50 cm/s',fontsize=20,fontweight='bold')

plt.clim(25,30)
ax.tick_params(labelcolor='k',labelsize=22,pad=15,grid_linewidth=2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=18)
cax.set_ylabel(ca,{'fontsize':32,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label='$\degree$C',cax=cax)
# =============================================================================
# 1d figure
# =============================================================================
#plt.params


    #label 
    
    # plt.savefig(savedir+title_time[:10],dpi=200)



fig, ax = plt.subplots(figsize=(7.5,6.5 ),linewidth=5)
ax.plot(map_data[:,0],map_data[:,1],color='k')

Q = ax.quiver(Lon,Lat, Uwind_kuri_S1C1 - Uwind_kuri_S1Cr, Vwind_kuri_S1C1 - Vwind_kuri_S1Cr,units='xy',
              scale=5,color='r',width=0.10,zorder = 1)    # width=0.003, scale=5000,

Q1 = ax.quiver(Lon[0,0],Lat[0,0],Uwind[0,0],Vwind[0,0],units='xy',scale=5,
               color='r',width=0.15,zorder = 1)

ax.quiverkey(Q1, X = 0.8, Y = -0.1,  U = 10, label='', labelpos="E")  
ax.text(139.5,16.5,'10 knot',fontsize=20,fontweight='bold')

plt.yticks(np.arange(10,51,5))
ax.set_yticklabels(np.arange(10,51,5),fontweight='bold',fontsize=13)
plt.xticks(np.arange(110,161,5))
ax.set_xticklabels(np.arange(110,161,5),fontweight='bold',fontsize=13)














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

        
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       










