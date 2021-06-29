#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 15:51:35 2021

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


r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/ERA5/' 
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA/'


Time = ['1993-01','2019-12'] 
minlon,maxlon = 112,260
minlat,maxlat = -10,70,
# mindepth, maxdepth = 0, 1000
    

# data = xr.open_mfdataset(r_path+'*.nc', parallel=True)

Sample_Data1 = xr.open_dataset(r_path1+'ERA5_landSea_mask.nc')
Sample_Data2 = xr.open_dataset(r_path1+'ERA5_single_level.nc')

data = xr.merge([Sample_Data1,Sample_Data2])

# data_s = data.where( (data.lat>=minlat)&(data.lat<=maxlat)&(data.lon>=minlon)&(data.lon<=maxlon),
#                     drop=True)
# data_s = data.loc[dict(latitude=slice(maxlat,minlat),
#                        longitude=slice(minlon,maxlon),expver=1 )]
data_s = data.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(maxlat,minlat),
                        longitude=slice(minlon,maxlon),expver=1 )]

data_s = data_s.where(data_s.lsm==0,drop=False)

data_M = data_s.mean(dim='time')



# =============================================================================
# tmp Check (Mask)
# =============================================================================

tmp = data_s.lsm.mean(dim='time')
n1,n2,n3,n4,n5 = 0,0,0,0,0
for i in tqdm(range(321)):
    for j in range(593):
        # if (tmp[i,j] != 1) and (tmp[i,j] != 0):
        #     n1+=1
        #     print(tmp[i,j].values)
        #     print(i,j)
        if tmp[i,j] == 1:
            n2+=1
        elif tmp[i,j] == 0:
            n3+=1
        elif 0<tmp[i,j]<=.5 :
            n4+=1
        elif .5<tmp[i,j]<1:
            n5+=1
        else :
            print('SS')    
            
print('n1 = '+str(n1))
print('n2 = '+str(n2))
print('n3 = '+str(n3))
print('n4 = '+str(n4))
print('n5 = '+str(n5))

plt.contour(data_M.lsm.values,colors='#72f542')

# =============================================================================
# Fig params
# =============================================================================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['font.size'] = 24


# =============================================================================
# TMP matlab CurlZ
# =============================================================================

tmp_data1 = data_s.u10.values
tmp_data2 = data_s.v10.values
tmp_data3 = data_s.latitude.values
tmp_data4 = data_s.longitude.values
tmp_path = '/home/caelus/dock_1/Working_hub/DATA_dep/tmp/'
np.save(tmp_path+'u10_s',tmp_data1)
np.save(tmp_path+'v10_s',tmp_data2)
np.save(tmp_path+'lat',tmp_data3)
np.save(tmp_path+'lon',tmp_data4)

curlZ_M = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/tmp/curlZ_s.npy') 


# =============================================================================
# Mean data / fig [ 0 ]
# =============================================================================

figdata01 = np.flipud(data_M.msl.values)/100 # Pa --> hpa
figdata02 = np.flipud(np.mean(curlZ_M,axis=0))
lon_00 = data_M.longitude.values  
lat_00 = np.flipud(data_M.latitude.values)

max_figdata01, min_figdata01 = np.nanmax(figdata01), np.nanmin(figdata01)
max_figdata02, min_figdata02 = np.nanmax(figdata02), np.nanmin(figdata02)

# --------------------------figure-------------------------------------

# w_path01 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/'


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
lon_m, lat_m = np.meshgrid(lon_00,lat_00)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title(' Climatological WSC & Pressure ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m,lat_m,figdata01,colors='grey',linewidths=2.5,levels=10)
plt.clim(min_figdata01,max_figdata01)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_00,lat_00,figdata02*10**7,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-1,1)
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




# =============================================================================
# Mean data / fig [ 1 ]
# =============================================================================

figdata11 = data_s.msl.values/100 # Pa --> hpa
figdata12 = curlZ_s
lon_10 = data_s.longitude.values  
lat_10 = np.flipud(data_s.latitude.values)
lon_m10, lat_m10 = np.meshgrid(lon_10,lat_10)

max_figdata11, min_figdata11 = np.nanmax(figdata11), np.nanmin(figdata11)
# max_figdata12, min_figdata12 = np.nanmax(figdata12), np.nanmin(figdata12)


Sig_set,Corr_map,Annual_mean = sig_pro(r_path2,['1993-01-01',324,300],Standard=True)



Sig_set.keys()

Sig = Sig_set.PDO_2Y_Rm.dropna()[:-1]

Coef_WSC1, _ = linearRegress4Cube(Sig,figdata12[12:-12,:,:],['1994-01','2018-12'])
Coef_WSC2, _ = linearRegress4Cube(Sig,figdata12[12:-12,:,:],['1994-01','2005-10'])
Coef_WSC3, _ = linearRegress4Cube(Sig,figdata12[12:-12,:,:],['2005-11','2011-01'])
Coef_WSC4, _ = linearRegress4Cube(Sig,figdata12[12:-12,:,:],['2005-11','2013-03'])
Coef_WSC5, _ = linearRegress4Cube(Sig,figdata12[12:-12,:,:],['2013-04','2018-12'])

Coef_msl1, _ = linearRegress4Cube(Sig,figdata11[12:-12,:,:],['1994-01','2018-12'])
Coef_msl2, _ = linearRegress4Cube(Sig,figdata11[12:-12,:,:],['1994-01','2005-10'])
Coef_msl3, _ = linearRegress4Cube(Sig,figdata11[12:-12,:,:],['2005-11','2011-01'])
Coef_msl4, _ = linearRegress4Cube(Sig,figdata11[12:-12,:,:],['2005-11','2013-03'])
Coef_msl5, _ = linearRegress4Cube(Sig,figdata11[12:-12,:,:],['2013-04','2018-12'])

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_WSC_199401_201812',Coef_WSC1)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_WSC_199401_200510',Coef_WSC2)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_WSC_200511_201101',Coef_WSC3)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_WSC_200511_201303',Coef_WSC4)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_WSC_201304_201812',Coef_WSC5)

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_msl_199401_201812',Coef_msl1)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_msl_199401_200510',Coef_msl2)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_msl_200511_201101',Coef_msl3)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_msl_200511_201303',Coef_msl4)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/data/Coef_PDO_msl_201304_201812',Coef_msl5)



# --------------------------figure-------------------------------------

w_path11 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA5/figs/'


# 1
fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title(' WSC & P reg (PDO) 2013-04 ~ 2018-12 ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m10,lat_m10,np.flipud(Coef_msl5),colors='grey',linewidths=2.5,levels=10)
plt.clim(-.01,.01)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(Coef_WSC5*10**7),cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.2,.2)
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
h = plt.colorbar(label='$10^{7}$ [factor]',cax=cax);
plt.savefig('/home/caelus/wormhole/tmp4ppt/WSC_P_regressed_PDO_201304_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()



# 2
fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title(' WSC & P reg (YS) 1994-01 ~ 2005-10 ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m10,lat_m10,np.flipud(Coef_msl2),colors='grey',linewidths=2.5,levels=10)
plt.clim(min_figdata01,max_figdata01)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(Coef_WSC2*10**7),cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.2,.2)
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
h = plt.colorbar(label='$10^{7}$ [factor]',cax=cax);
plt.savefig(w_path11+'WSC_P_regressed_YS_199401_200510',bbox_inches='tight')
plt.tight_layout()
plt.show()


# 3
fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title(' WSC & P reg (YS) 2005-11 ~ 2011-01 ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m10,lat_m10,np.flipud(Coef_msl3),colors='grey',linewidths=2.5,levels=10)
plt.clim(min_figdata01,max_figdata01)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(Coef_WSC3*10**7),cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.2,.2)
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
h = plt.colorbar(label='$10^{7}$ [factor]',cax=cax);
plt.savefig(w_path11+'WSC_P_regressed_YS_200511_201101',bbox_inches='tight')
plt.tight_layout()
plt.show()


# 4
fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title(' WSC & P reg (YS) 2005-11 ~ 2013-03 ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m10,lat_m10,np.flipud(Coef_msl4),colors='grey',linewidths=2.5,levels=10)
plt.clim(min_figdata01,max_figdata01)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(Coef_WSC4*10**7),cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.2,.2)
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
h = plt.colorbar(label='$10^{7}$ [factor]',cax=cax);
plt.savefig(w_path11+'WSC_P_regressed_YS_200511_201303',bbox_inches='tight')
plt.tight_layout()
plt.show()


# 5
fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title(' WSC & P reg (YS) 2013-03 ~ 2018-12 ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m10,lat_m10,np.flipud(Coef_mls5),colors='grey',linewidths=2.5,levels=10)
plt.clim(min_figdata01,max_figdata01)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(Coef_WSC5*10**7),cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.2,.2)
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
h = plt.colorbar(label='$10^{7}$ [factor]',cax=cax);
plt.savefig(w_path11+'WSC_P_regressed_YS_201303_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()






