#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 17:48:41 2021

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
# ADT & EKE
# =============================================================================

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Detrend/data/'

Data = xr.open_mfdataset(r_path1+'*.nc', parallel=True)

Data = Data.loc[dict(longitude=slice(112,260),latitude=slice(0,60))] 

EKE = Data.EKE.loc[dict(longitude=slice(120,180),latitude=slice(18,30))] 

n = 1
# ------ Running mean ------
Data = Data.fillna(-999)
WD = n*12
Data_nY = Data.rolling(time=WD,center=True).mean().dropna("time")
Data_nY = Data_nY.where(Data_nY!=-999,drop=False) 

EKE = EKE.fillna(-999)
EKE_nY = EKE.rolling(time=WD,center=True).mean().dropna("time")
EKE_nY = EKE_nY.where(EKE_nY!=-999,drop=False) 

# ------ Def fig Variables ------

lat11 = Data_nY.latitude.values
lon11 = Data_nY.longitude.values 
figdata11 = Data_nY.adt.values
lon_m11, lat_m11 = np.meshgrid(lon11,lat11)


lat21 = EKE_nY.latitude.values
lon21 = EKE_nY.longitude.values 
figdata21 = EKE_nY.values
lon_m21, lat_m21 = np.meshgrid(lon21,lat21)


# ------------ test fig ADT ----------------

fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=60,\
            llcrnrlon=112,urcrnrlon=260,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')

cs2 = m.pcolormesh(lon_m11,lat_m11,figdata11[0,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-1.5,1.5)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='10 [factor]',cax=cax);
# plt.savefig(w_path21+'adt_regressed_YS_201304_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()


# ------------ test fig EKE ----------------


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=19,urcrnrlat=27,\
            llcrnrlon=120,urcrnrlon=180,resolution='i')
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
cs2 = m.pcolormesh(lon_m21,lat_m21,figdata21[100,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-2,2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='100 [factor]',cax=cax);
# plt.savefig(w_path21+'adt_regressed_YS_201304_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()



# =============================================================================
# Signal
# =============================================================================
r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300])

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')




# =============================================================================
# =============================================================================
# # Total figure
# =============================================================================
# =============================================================================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

w_path_sig = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Detrend/Figure_RM/'
n = 12
while n < 312:  

# ------------ fig ADT ----------------
    fig, ax = plt.subplots(figsize=(16,7),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=60,\
                llcrnrlon=112,urcrnrlon=260,resolution='i')
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    plt.title('a) Date : '+Sig_set.dates[n] + ' [ADTa(dt) 1Y filtered]', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')

    m.plot([120,180,180,120,120],[18,18,28,28,18],color='k',linestyle='--',linewidth=4,alpha=.8)
    
    cs2 = m.pcolormesh(lon_m11,lat_m11,figdata11[n-12,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    plt.clim(-1.5,1.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='$10^{-1} [m]$',cax=cax);
    plt.tight_layout()
    plt.savefig(w_path_sig+'/adt_1Y/ADTa_'+Sig_set.dates[n])
    plt.show()


# ------------ fig EKE ----------------

    fig, ax = plt.subplots(figsize=(16,3),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=19,urcrnrlat=27,\
                llcrnrlon=120,urcrnrlon=180,resolution='i')
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,2.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,5.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    plt.title('b) Date : '+Sig_set.dates[n] + ' [EKEa(dt) 1Y filtered]', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs = m.pcolormesh(lon_m,lat_m,data)
    # cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    
    cs2 = m.pcolormesh(lon_m21,lat_m21,figdata21[n-12,:,:]*1000,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.clim(-max_figdata02,max_figdata02)
    plt.clim(-200,200)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='$10 [(cm/s)^{2}$]',cax=cax);
    plt.tight_layout()
    plt.savefig(w_path_sig+'/EKE_1Y/ADTa_'+Sig_set.dates[n])
    plt.show()
    n+=1





























