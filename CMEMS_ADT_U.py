#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:04:18 2021

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

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/CMEMS/' 
r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/'

# w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA/'

# nc_list = np.sort([file for file in os.listdir(r_path1) if file.endswith(".nc")])

# Sample_Data1 = xr.open_dataset(r_path1+nc_list[0])




Sig_set,Corr_map,Annual_mean = sig_pro(r_path1,['1993-01-01',324,300],Standard=True)

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')

Sig = Sig_set.ADT_index_2Y_Rm



data_CMEMS = xr.open_mfdataset(r_path2+'*.nc', parallel=True)
# data_CDS_adt = xr.open_dataset(r_path3+'adt_0_60_112_260_M.nc')
data_CDS_u = xr.open_dataset(r_path3+'ugos_0_42_110_260_M.nc')
data_CDS_v = xr.open_dataset(r_path3+'vgos_0_42_110_140_M.nc')

data1 = data_CDS_u.ugos.values
data2 = data_CDS_v.vgos.values

CDS_U_coef1, _ = linearRegress4Cube(Sig,data1,['1994-01','2018-12'])
CDS_U_coef2, _ = linearRegress4Cube(Sig,data1,['1994-01','2005-10'])
CDS_U_coef3, _ = linearRegress4Cube(Sig,data1,['2005-11','2011-01'])
CDS_U_coef4, _ = linearRegress4Cube(Sig,data1,['2005-11','2013-03'])
CDS_U_coef5, _ = linearRegress4Cube(Sig,data1,['2013-04','2018-12'])

CDS_V_coef1, _ = linearRegress4Cube(Sig,data2,['1994-01','2018-12'])
CDS_V_coef2, _ = linearRegress4Cube(Sig,data2,['1994-01','2005-10'])
CDS_V_coef3, _ = linearRegress4Cube(Sig,data2,['2005-11','2011-01'])
CDS_V_coef4, _ = linearRegress4Cube(Sig,data2,['2005-11','2013-03'])
CDS_V_coef5, _ = linearRegress4Cube(Sig,data2,['2013-04','2018-12'])


np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_199401_201812',CDS_U_coef1)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_199401_200510',CDS_U_coef2)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_200511_201101',CDS_U_coef3)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_200511_201303',CDS_U_coef4)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_201304_201812',CDS_U_coef5)

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_199401_201812_140',CDS_V_coef1)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_199401_200510_140',CDS_V_coef2)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_200511_201101_140',CDS_V_coef3)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_200511_201303_140',CDS_V_coef4)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_201304_201812_140',CDS_V_coef5)

plt.pcolormesh(CDS_U_coef1,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-.1,.1)
plt.colorbar()

plt.pcolormesh(Ctemp_surface1,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()

plt.pcolormesh(Ctemp_surface1,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()

plt.pcolormesh(Ctemp_surface1,cmap=plt.cm.get_cmap('seismic'))
plt.clim(-3,3)
plt.colorbar()

# =============================================================================
# 
# =============================================================================

data_CDS_u = xr.open_dataset(r_path3+'ugos_0_42_110_260_M.nc')
data_CDS_v = xr.open_dataset(r_path3+'vgos_0_42_110__M.nc')

data1 = data_CDS_u.ugos.values
data2 = data_CDS_v.vgos.values

lat10 = data_CDS_u.lat.values
lon10 = data_CDS_u.lon.values 

lon_m10, lat_m10 = np.meshgrid(lon10,lat10)

figdata111 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_199401_201812.npy')
figdata112 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_199401_200510.npy')
figdata113 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_200511_201101.npy')
figdata114 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_200511_201303.npy')
figdata115 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_ugos_201304_201812.npy')

figdata121 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_199401_201812.npy')
figdata122 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_199401_200510.npy')
figdata123 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_200511_201101.npy')
figdata124 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_200511_201303.npy')
figdata125 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/data/Coef_YS_vgos_201304_201812.npy')


w_path1 ='/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CDS_CMEMS/figs/' 

# ------------ test fig ADT ----------------

fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=260,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed U 1994-01 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata111,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'u_regressed_YS_199301_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=260,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed U 1994-01 ~ 2005-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata112,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'u_regressed_YS_199301_200503',bbox_inches='tight')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=260,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed U 2005-03 ~ 2011-01  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata113,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'u_regressed_YS_200504_201101',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=260,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed U 2005-04 ~ 2013-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata114,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'u_regressed_YS_200504_201303',bbox_inches='tight')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=260,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed U 2013-04 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata115,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'u_regressed_YS_201304_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()

# V


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=140,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed V \n1994-01 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=28,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata121,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'v_regressed_YS_199301_201812_140',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=140,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed V \n1994-01 ~ 2005-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=28,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata122,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'v_regressed_YS_199301_200503_140',bbox_inches='tight')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=140,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed V \n2005-03 ~ 2011-01  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=28,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata123,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'v_regressed_YS_200504_201101_140',bbox_inches='tight')
plt.tight_layout()
plt.show()



fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=140,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed V \n2005-04 ~ 2013-03  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=28,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata124,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'v_regressed_YS_200504_201303_140',bbox_inches='tight')
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=-10,urcrnrlat=42,\
            llcrnrlon=110,urcrnrlon=140,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title('Regressed V \n2013-04 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=28,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata125,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
plt.savefig(w_path1+'v_regressed_YS_201304_201812_140',bbox_inches='tight')
plt.tight_layout()
plt.show()






# =============================================================================
# 
# =============================================================================


Time = ['1994-01','2018-12'] 
# minlon,maxlon = 112,180
fixed_lon = 150
minlat,maxlat = 5,45
mindepth, maxdepth = 0, 200
    
data_CMEMS_s = data_CMEMS.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(minlat,maxlat),depth=slice(mindepth,maxdepth))]
data_CDS_s = data_CDS.loc[dict(time=slice(Time[0],Time[1]),y=slice(minlat,maxlat))]




temp_surface = data_s.thetao.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(minlat,maxlat),depth=slice(mindepth,0.5))].squeeze()
temp = data_s.thetao


data1 = temp_surface.values




# =============================================================================
# 
# =============================================================================















