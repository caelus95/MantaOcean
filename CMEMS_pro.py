#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 18:56:36 2021

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

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/CMEMS/' 
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

# w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA/'

nc_list = np.sort([file for file in os.listdir(r_path1) if file.endswith(".nc")])


Sample_Data1 = xr.open_dataset(r_path1+nc_list[0])

data = xr.open_mfdataset(r_path1+'*.nc', parallel=True)
data_M = data.mean(dim='time')


Time = ['1994-01','2018-12'] 
minlon,maxlon = 112,180
fixed_lon = 150
minlat,maxlat = 5,45
mindepth, maxdepth = 0, 2300
    

data_s = data.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(minlat,maxlat),
                        longitude=fixed_lon ,depth=slice(mindepth,maxdepth))]

data_s_M = data_s.mean(dim='time')




# =============================================================================
# 
# =============================================================================

lat_M = data_M.latitude.values
lon_M = data_M.longitude.values
lon_mesh_M,lat_mesh_M = np.meshgrid(lon,lat)

mapdata_1 = data_M.zos.values
max_mapdata_1, min_mapdata_1 = np.max(mapdata_1), np.min(mapdata_1)
# =============================================================================
# 
# =============================================================================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "light"
plt.rcParams['axes.linewidth'] = 1
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True


# =============================================================================
# vertical Mean data
# =============================================================================
lat_s_M = data_s_M.latitude.values
lon_s_M = data_s_M.longitude.values
depth_s_M = -np.flipud(data_s_M.depth)
lat_mesh_s_M,depth_mesh_s_M = np.meshgrid(lat_s_M,depth_s_M)
mapdata_1 = np.flipud(data_s_M.uo.values)
max_mapdata_1, min_mapdata_1 = np.nanmax(mapdata_1), np.nanmin(mapdata_1)
# clim_val = max(max_mapdata_2,min_mapdata_2)

from gsw.density import rho 

so = data_s_M.so.values
thetao = data_s_M.thetao.values
press = data_s_M.depth.values


den=np.zeros_like(so)
for i in tqdm(range(481)):
    den[:,i] = rho(so[:,i],thetao[:,i],press)

den = np.flipud(den)
# =============================================================================
# vertical data
# ==========================================================================================================================================================
lat_s = data_s.latitude.values
lon_s = data_s.longitude.values
depth_s = -np.flipud(data_s.depth)
lat_mesh_s,depth_mesh_s = np.meshgrid(lat_s,depth_s)
press_s = data_s.depth

uo_s =data_s.uo.values
thetao_s =data_s.thetao.values
so_s =data_s.so.values


from gsw.density import rho 

t,at,on = so_s.shape
den_s=np.zeros_like(so_s)
for i in tqdm(range(t)):
    for j in range(481):
        den_s[i,:,j] = np.flipud(rho(so_s[i,:,j],thetao_s[i,:,j],press_s))

# Sig_sets
Sig_sets,Corr_map,Annual_mean = sig_pro(r_path2,['1993-01-01',324,300],Standard=True)
Sig = Sig_sets.ADT_index_2Y_Rm.dropna()[:-1]


# Regression
time_slice = ['2013-04','2018-12']
uo_Coef, _ = linearRegress4Cube(Sig,uo_s,[time_slice[0],time_slice[1]])
den_Coef, _ = linearRegress4Cube(Sig,den_s,[time_slice[0],time_slice[1]])
np.save('/home/caelus/wormhole/PPP/uo_Coef'+str(time_slice[0])+'_'+str(time_slice[1]),uo_Coef)
np.save('/home/caelus/wormhole/PPP/den_Coef'+str(time_slice[0])+'_'+str(time_slice[1]),den_Coef)


tmp_r_path = '/home/caelus/wormhole/PPP/'
mapdata_2 = np.load(tmp_r_path+'uo_Coef1994-01_2018-12.npy')
mapdata_3 = np.load(tmp_r_path+'den_Coef1994-01_2018-12.npy')

mapdata_4 = np.load(tmp_r_path+'uo_Coef1994-01_2005-10.npy')
mapdata_5 = np.load(tmp_r_path+'den_Coef1994-01_2005-10.npy')

mapdata_6 = np.load(tmp_r_path+'uo_Coef2005-11_2011-01.npy')
mapdata_7 = np.load(tmp_r_path+'den_Coef2005-11_2011-01.npy')

# =============================================================================
# Vertical plot 1
# =============================================================================

plt.figure(figsize=(15,11))
ax1=plt.gca()
plt.ylim(-900,0)
plt.yticks(fontsize=22, alpha=1,)
plt.ylabel('$\it{depth(m)}$',fontsize=24,fontweight='light')
plt.xticks(fontsize=22, alpha=1)
plt.xlabel('$\it{latitude (\degree N)}$',fontsize=24,fontweight='light')
plt.title('Resgress 1994~2018 ('+str(fixed_lon)+'$\degree$E) ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
# cs1 = plt.contour(depth_mesh_s,lat_mesh_s,mapdata_3,levels=10,colors='k',linestyles='-.',alpha=.8)
# plt.clabel(cs1,fontsize=10,fmt='%1.1f')

# plt.grid(axis='x', alpha=.5)
plt.axvline(x=17.5,color='k',linewidth=2,linestyle='--',alpha=.3)
plt.axvline(x=26.5,color='k',linewidth=2,linestyle='--',alpha=.3)

cs2 = plt.pcolormesh(lat_mesh_s,depth_mesh_s,uo_Coef,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_mapdata_2,max_mapdata_2)
plt.clim(-.3,.3)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label='m/s',cax=cax);
# plt.savefig('/home/caelus/wormhole/PPP/Mean_Vertical_'+str(fixed_lon)+'E',bbox_inches='tight')
plt.show()



# =============================================================================
# Horizontal data
# =============================================================================


data_uo_h = data.uo.loc[dict(depth=slice(15,150))]

uo_mean_150 = data_uo_h.mean(dim='time').values

depth_150 = data_uo_h.depth.values


def Manta_diff(x,y):
    tmp_dx = x[0:-1] - x[1:]
    tmp_dy = y[0:-1] - y[1:]
    return tmp_dx/tmp_dy

_,at_150,on_150 = uo_mean_150.shape

dudz = np.zeros([at_150,on_150])
for i in range(at_150):
    for j in range(on_150):
        dudz[i,j] = np.sum(Manta_diff(uo_mean_150[:,i,j],-depth_150))
        


lon_uo_150 = data_uo_h.longitude.values 
lat_uo_150 = data_uo_h.latitude.values 
mapdata_11 = dudz
max_mapdata_11, min_mapdata_11 = np.nanmax(mapdata_11), np.nanmin(mapdata_11)

# =============================================================================
# Horizontal data 2
# =============================================================================
time_slice = ['2013-04','2018-12']
tmp_dataset = data.zos.values[12:-12,:,:]
zos_Coef, _ = linearRegress4Cube(Sig,tmp_dataset,[time_slice[0],time_slice[1]])
np.save('/home/caelus/wormhole/PPP/zos_Coef'+str(time_slice[0])+'_'+str(time_slice[1]),uo_Coef)

# =============================================================================
# Horizontal map1 Mean
# =============================================================================
fig, ax = plt.subplots(figsize=(15,11),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=60,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=16,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=16,fontweight='bold',color='grey')
plt.title('$\mathit{Climatological}$'+' $\mathit{ SSH}$', 
          fontproperties='', position=(0.5, 1.0+0.05), fontsize=40,fontweight='bold')
# plt.suptitle(' (10 ~ 150db) ',fontstyle='italic',position=(0.5, .85),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.pcolormesh(lon_mesh_M,lat_mesh_M,mapdata_1,cmap=plt.cm.get_cmap('jet'))
plt.clim(-.3,1.3)

# cs2 = m.pcolormesh(lon_m,lat_m,mapdata_6,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(min_mapdata_6,max_mapdata_6)

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
h = plt.colorbar(label='$\mathit{m}$',cax=cax);
# plt.savefig('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/tmp_wind/Climatological_Ugdz',bbox_inches='tight')
plt.savefig('/home/caelus/wormhole/PPP/Climatological_zos',bbox_inches='tight')
plt.tight_layout()
plt.show()


# =============================================================================
# Horizontal map 2
# =============================================================================
fig, ax = plt.subplots(figsize=(15,11),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=60,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
lon_m, lat_m = np.meshgrid(lon_uo_150,lat_uo_150)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=16,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=16,fontweight='bold',color='grey')
plt.title('$\mathit{Climatological }$'+' $\mathit{\partial Ug/ \partial z}$', 
          fontproperties='', position=(0.5, 1.0+0.05), fontsize=40,fontweight='bold')
# plt.suptitle(' (10 ~ 150db) ',fontstyle='italic',position=(0.5, .85),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.pcolormesh(lon_m,lat_m,mapdata_11*10**2,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-1.5,1.5)

# cs2 = m.pcolormesh(lon_m,lat_m,mapdata_6,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(min_mapdata_6,max_mapdata_6)

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
h = plt.colorbar(label='$\mathit{\partial Ug/ \partial z }$'+' $\mathit{[10^{-2}s^{-1}]}$',cax=cax);
# plt.savefig('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/tmp_wind/Climatological_Ugdz',bbox_inches='tight')
plt.savefig('/home/caelus/wormhole/PPP/Climatological_Ugdz',bbox_inches='tight')
plt.tight_layout()
plt.show()

# =============================================================================
# Horizontal map 3
# =============================================================================
fig, ax = plt.subplots(figsize=(15,11),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=60,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
lon_m, lat_m = np.meshgrid(lon_uo_150,lat_uo_150)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=16,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=16,fontweight='bold',color='grey')
plt.title('$\mathit{Regressed }$'+' $\mathit{SSH}$', 
          fontproperties='', position=(0.5, 1.0+0.05), fontsize=40,fontweight='bold')
plt.suptitle(' (2013-04~2018-12) ',fontstyle='italic',position=(0.5, .913),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.pcolormesh(lon_m,lat_m,zos_Coef,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-.3,.3)

# cs2 = m.pcolormesh(lon_m,lat_m,mapdata_6,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(min_mapdata_6,max_mapdata_6)

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
h = plt.colorbar(label='',cax=cax);
# plt.savefig('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/tmp_wind/Climatological_Ugdz',bbox_inches='tight')
plt.savefig('/home/caelus/wormhole/PPP/Regressed_SSH_201304_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()










