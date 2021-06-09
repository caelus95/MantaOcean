#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 22:36:54 2021

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
# fixed_lon = 150
minlat,maxlat = 5,45
mindepth, maxdepth = 0, 200
    

data_s = data.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(minlat,maxlat),depth=slice(mindepth,maxdepth))]



temp_surface = data_s.thetao.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(minlat,maxlat),depth=slice(45,50))].squeeze()
temp = data_s.thetao


data1 = temp_surface.values


Sig_set,Corr_map,Annual_mean = sig_pro(r_path2,['1993-01-01',324,300],Standard=True)



Sig = Sig_set.ADT_index_2Y_Rm.dropna()[:-1]


Ctemp_surface1, _ = linearRegress4Cube(Sig,data1,['1994-01','2018-12'])
Ctemp_surface2, _ = linearRegress4Cube(Sig,data1,['1994-01','2005-10'])
Ctemp_surface3, _ = linearRegress4Cube(Sig,data1,['2005-11','2011-01'])
Ctemp_surface4, _ = linearRegress4Cube(Sig,data1,['2005-11','2013-03'])
Ctemp_surface5, _ = linearRegress4Cube(Sig,data1,['2013-04','2018-12'])


np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_199401_201812_50m',Ctemp_surface1)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_199401_200510_50m',Ctemp_surface2)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_200511_201101_50m',Ctemp_surface3)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_200511_201303_50m',Ctemp_surface4)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_201304_201812_50m',Ctemp_surface5)

# =============================================================================
# 
# =============================================================================

Ctemp_surface1 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_199401_201812_50m.npy')
Ctemp_surface2 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_199401_200510_50m.npy')
Ctemp_surface3 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_200511_201101_50m.npy')
Ctemp_surface4 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_200511_201303_50m.npy')
Ctemp_surface5 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/Coef_YS_adt_201304_201812_50m.npy')



plt.pcolormesh(Ctemp_surface1,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()


plt.pcolormesh(Ctemp_surface2,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()

plt.pcolormesh(Ctemp_surface3,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()

plt.pcolormesh(Ctemp_surface4,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()
plt.pcolormesh(Ctemp_surface5,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()



lat10 = temp.latitude.values
lon10 = temp.longitude.values

lon_m10, lat_m10 = np.meshgrid(lon10,lat10)
w_path1 ='/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/temp/' 


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed surface temp \n1994-01 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface1,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_199301_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed surface temp \n1994-01 ~ 2005-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface2,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_199301_200503',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed surface temp \n2005-03 ~ 2011-01  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface3,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_200503_201101',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed surface temp \n2005-03 ~ 2013-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface4,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_200503_201303',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed surface temp \n2013-04 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface5,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_201304_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()


# =============================================================================
# 50 m
# =============================================================================


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed 50m temp \n1994-01 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface1,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_199301_201812_50m',bbox_inches='tight')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed 50m temp \n1994-01 ~ 2005-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface2,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_199301_200503_50m',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed 50m temp \n2005-03 ~ 2011-01  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface3,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_200503_201101_50m',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed 50m temp \n2005-03 ~ 2013-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface4,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_20503_201303_50m',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=5,urcrnrlat=45,\
            llcrnrlon=112,urcrnrlon=180,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed 50m temp \n2013-04 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,Ctemp_surface5,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3,3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'temp_surface_regressed_YS_201304_201812_50m',bbox_inches='tight')
plt.tight_layout()
plt.show()

