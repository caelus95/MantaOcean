#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:41:19 2021

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




ADT_t = xr.open_dataset('/home/caelus/dock_1/Working_hub/DATA_dep/CDS/T_CDS_monthly_199301_201912.nc',decode_times=False)

minlon,maxlon = 112,260
minlat,maxlat = 0,60,

data_s = ADT_t.loc[dict(latitude=slice(minlat,maxlat),
                        longitude=slice(minlon,maxlon),nv= 1 )]



# define a function to compute a linear trend of a timeseries
def linear_trend(x):
    pf = np.polyfit(x.time, x, 1)
    # need to return an xr.DataArray for groupby
    return xr.DataArray(pf[0])



data_ADT_a = data_s - data_s.mean(dim='time')


# stack lat and lon into a single dimension called allpoints
stacked = data_ADT_a.adt.stack(allpoints=['latitude','longitude'])
# apply the function over allpoints to calculate the trend at each point
trend = stacked.groupby('allpoints').apply(linear_trend)
# unstack back to lat lon coordinates
trend_unstacked = trend.unstack('allpoints')

data_ADT_a_dt = data_ADT_a.adt - trend_unstacked

data_ADT_a_dt = data_ADT_a_dt.fillna(-999)

WD = 2*12
data_ADT_a_dt_2Y = data_ADT_a_dt.rolling(time=WD,center=True).mean().dropna("time")
data_ADT_a_dt_2Y = data_ADT_a_dt_2Y.where(data_ADT_a_dt_2Y!=-999,drop=False) 

lat30 = data_ADT_a_dt_2Y.latitude.values
lon30 = data_ADT_a_dt_2Y.longitude.values 
figdata30 = data_ADT_a_dt_2Y.values

lon_m30, lat_m30 = np.meshgrid(lon30,lat30)


r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300])

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')



# =============================================================================
# Figure
# =============================================================================

w_path_sig ='/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/Figures/SUM/detrend_ADTa/'

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True



n = 12
while n < 312:  
    fig, ax = plt.subplots(figsize=(16,7),linewidth=1)
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
    plt.title('a) Date : '+Sig_set.dates[n] + ' (ADTa 2Y filtered)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs = m.pcolormesh(lon_m,lat_m,data)
    # cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    
    m.plot([120,180,180,120,120],[18,18,28,28,18],color='k',linestyle='--',linewidth=4,alpha=.8)
    
    cs2 = m.pcolormesh(lon_m30,lat_m30,figdata30[n-12,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.clim(-max_figdata02,max_figdata02)
    plt.clim(-1.5,1.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='10 [factor]',cax=cax);
    plt.savefig(w_path_sig+'detrend_ADTa_'+Sig_set.dates[n])
    plt.tight_layout()
    plt.show()
    n+=1
    




fig, ax = plt.subplots(figsize=(16,7),linewidth=1)
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
plt.title('a) SSH Trend (1993-01~2019-12)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

m.plot([120,180,180,120,120],[18,18,28,28,18],color='k',linestyle='--',linewidth=4,alpha=.8)

cs2 = m.pcolormesh(lon_m30,lat_m30,trend_unstacked*10**5,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-3.,3.)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='$10^{-5}$'+' [m/month]',cax=cax);
# plt.savefig(w_path_sig+'/ADTa/ADTa_'+Sig_set.dates[n])
plt.tight_layout()
plt.show()

    


































