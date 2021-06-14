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
# Sig_set
# =============================================================================

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path1,['1993-01-01',324,300],Standard=True)

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')


# =============================================================================
# WSC & Press
# =============================================================================

r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/ERA5/' 


Time = ['1993-01','2019-12'] 
minlon,maxlon = 112,260
minlat,maxlat = -10,70,
    

Sample_Data1 = xr.open_dataset(r_path2+'ERA5_landSea_mask.nc')
Sample_Data2 = xr.open_dataset(r_path2+'ERA5_single_level.nc')

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

# =============================================================================
# Def r_vector
# r_vectors
# =============================================================================
# x,y = data_WSC_a_2Y.longitude, data_WSC_a_2Y.latitude
# data1, data2 = data_WSC_a_2Y.u10, data_WSC_a_2Y.v10
# factor = [6, 6]

def r_vector4cube(x,y,data1,data2,factor):
    xx = x.values.shape[0]
    yy = y.values.shape[0]
    a = np.arange(0,xx,factor[0])
    b = np.arange(0,yy,factor[1])
    
    r_x, r_y = x[a], y[b]
    
    r_data1 = data1.where( (data1.longitude==r_x) & (data1.latitude==r_y), drop=True )
    r_data2 = data2.where( (data2.longitude==r_x) & (data2.latitude==r_y), drop=True )    
    
    return r_x, r_y, r_data1, r_data2

x,y = data_WSC_a_2Y.longitude, data_WSC_a_2Y.latitude
data1, data2 = data_WSC_a_2Y.u10, data_WSC_a_2Y.v10
factor = [6, 6]

r_x, r_y, r_data1, r_data2 = r_vector4cube(x,y,data1,data2,[10,10])



# =============================================================================
# ADT 
# =============================================================================


# ADT
r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/'
ADT = xr.open_dataset(r_path3+'adt_10_70_112_260_M.nc')

adt = ADT.adt

data_adt_a = adt - adt.mean(dim='time')

data_adt_a = data_adt_a.fillna(-999)

WD = 2*12
data_adt_a_2Y = data_adt_a.rolling(time=WD,center=True).mean().dropna("time")
data_adt_a_2Y = data_adt_a_2Y.where(data_adt_a_2Y!=-999,drop=False)


# test plot for WSC, msl, UV

lat11, lon11 = np.flipud(data_WSC_a_2Y.latitude.values), data_WSC_a_2Y.longitude.values 
lat12, lon12 = np.flipud(r_y), r_x
lat13, lon13 = ADT.lat.values, ADT.lon.values

lon_m11, lat_m11 = np.meshgrid(lon11,lat11)
lon_m12, lat_m12 = np.meshgrid(lon12,lat12)
lon_m13, lat_m13 = np.meshgrid(lon13,lat13)

# Cube data [t, at, on] (flipud needed)
figdata111 = data_WSC_a_2Y.msl.values/100 
figdata112 = data_WSC_a_2Y.curlZ.values

figdata121 = r_data1.values
figdata122 = r_data2.values

figdata131 = data_adt_a_2Y.values

# ------------ test fig WSC ----------------

tmp_n = 0
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

# cs1 = m.contour(lon_m11,lat_m11,np.flipud(figdata111[tmp_n,:,:]),colors='grey',linewidths=2.5,levels=10)
# plt.clim(-3.3,3.3)
# plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m11,lat_m11,np.flipud(figdata112[tmp_n,:,:])*10**7,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-.3,.3) # plt.clim(-max_figdata02,max_figdata02)

# m.quiver(lon_m12,lat_m12,np.flipud(figdata121[tmp_n,:,:]),np.flipud(figdata122[tmp_n,:,:]),
#          headwidth=5,headaxislength=10,headlength=10)
q = m.quiver(lon_m12,lat_m12,np.flipud(figdata121[tmp_n,:,:]),np.flipud(figdata122[tmp_n,:,:]),
         scale=20,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25],
         minlength=1)
plt.axis('equal')
# Unit vector
p = plt.quiverkey(q,120,72,1,"1 m/s",coordinates='data',color='r',labelpos='E')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='$\mathit{10^{-7} N \cdot m^{-3}}$',cax=cax);
# plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
plt.tight_layout()
plt.show()


# =============================================================================
# loop 
# =============================================================================

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True


w_path_sig = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_WSC_UV/'
n = 12
while n < 312:  
    fig, ax = plt.subplots(figsize=(16,8.5),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
                llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
    # lon_m, lat_m = np.meshgrid(lon_00,lat_00)
    # x, y = m(lon, lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    plt.title('a) Date : '+Sig_set.dates[n] + ' (ADTa & UVa 2Y filtered)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs1 = m.contour(lon_m11,lat_m11,np.flipud(figdata111[n,:,:]),colors='grey',linewidths=2.5,levels=10)
    # plt.clim(-3.3,3.3)
    # plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

    cs2 = m.pcolormesh(lon_m13,lat_m13,figdata131[n-12,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    plt.clim(-1.5,1.5) #plt.clim(-.3,.3) # plt.clim(-max_figdata02,max_figdata02)
    
    q = m.quiver(lon_m12,lat_m12,np.flipud(figdata121[n-12,:,:]),np.flipud(figdata122[n-12,:,:]),
         scale=20,headwidth=7.5,headaxislength=10,headlength=13,color='k',
         minlength=1,edgecolor='y',minshaft=1.3,alpha=.7)
    # plt.axis('equal')
    # Unit vector
    p = plt.quiverkey(q,120,67,1,"1 m/s",coordinates='data',color='r',
                      labelpos='S',labelcolor='W',fontproperties={'size':16},
                      labelsep=0.13,alpha=1)
    
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='',cax=cax);
    h = plt.colorbar(label='10 [factor]',cax=cax);
    # plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(w_path_sig+'/adt_g/ADTa_UVa_'+Sig_set.dates[n])
    plt.show()
    n+=1
# -----------------------------
# n = 12
# while n < 312:  
#     fig, ax = plt.subplots(figsize=(16,8.5),linewidth=1)
#     ax = plt.gca()
#     m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
#                 llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
#     # lon_m, lat_m = np.meshgrid(lon_00,lat_00)
#     # x, y = m(lon, lat)
#     m.fillcontinents(color='black',lake_color='black')
#     m.drawcoastlines()
#     m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
#                     dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
#     m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
#                     dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
#     plt.title('a) Date : '+Sig_set.dates[n] + ' (WSCa & Pressa 2Y filtered)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
#     #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
#     cs1 = m.contour(lon_m11,lat_m11,np.flipud(figdata111[n-12,:,:]),colors='k',
#                     linewidths=2.5,levels=10,alpha=.7)
#     plt.clim(-3.3,3.3)
#     plt.clabel(cs1,fontsize=10,fmt='%1.2f',colors='k')

#     cs2 = m.pcolormesh(lon_m11,lat_m11,np.flipud(figdata112[n-12,:,:])*10**7,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
#     plt.clim(-.3,.3) # plt.clim(-max_figdata02,max_figdata02)
    
#     # q = m.quiver(lon_m12,lat_m12,np.flipud(figdata121[n-12,:,:]),np.flipud(figdata122[n-12,:,:]),
#     #      scale=20,headwidth=7.5,headaxislength=10,headlength=13,color='k',
#     #      minlength=1,edgecolor='y',minshaft=1.3,alpha=.7)
#     # plt.axis('equal')
#     # # Unit vector
#     # p = plt.quiverkey(q,120,67,1,"1 m/s",coordinates='data',color='r',
#     #                   labelpos='S',labelcolor='W',fontproperties={'size':16},
#     #                   labelsep=0.13,alpha=1)
    
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="2.5%", pad=0.1)
#     cax.tick_params(labelsize=15)
#     cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#     #label 
#     h = plt.colorbar(label='',cax=cax);
#     h = plt.colorbar(label='$\mathit{10^{-7} N \cdot m^{-3}}$',cax=cax);
#     # plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
#     plt.tight_layout()
#     plt.savefig(w_path_sig+'/3/WCSa_Pressa'+Sig_set.dates[n])
#     plt.show()
#     n+=1

















