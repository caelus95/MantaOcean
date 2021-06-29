#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 17:12:49 2021

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
import warnings
warnings.filterwarnings("ignore")

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/CDS/'

def linear_trend(x):
    pf = np.polyfit(x.time, x, 1)
    # need to return an xr.DataArray for groupby
    return xr.DataArray(pf[0])

def r_vector4cube(x,y,data1,data2,factor):
    xx = x.values.shape[0]
    yy = y.values.shape[0]
    a = np.arange(0,xx,factor[0])
    b = np.arange(0,yy,factor[1])
    
    r_x, r_y = x[a], y[b]
    
    r_data1 = data1.where( (data1.longitude==r_x) & (data1.latitude==r_y), drop=True )
    r_data2 = data2.where( (data2.longitude==r_x) & (data2.latitude==r_y), drop=True )    
    
    return r_x, r_y, r_data1, r_data2 

# ugos = xr.open_dataset(r_path+'ugos_0_42_110_260_M.nc') 
# vgos = xr.open_dataset(r_path+'vgos_0_42_110_260_M.nc') 

Data = xr.open_mfdataset(r_path+'*.nc', parallel=True,decode_times=False)

minlon,maxlon = 112,260
minlat,maxlat = 5,60,

data_s = Data.loc[dict(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),nv= 1 )]

##############
# mask for UV
mask_mlon,mask_Mlon = 130,175
mask_mlat,mask_Mlat = 30, 40

tmp_adt = data_s.adt 

data_s =data_s.where( (data_s.longitude<mask_mlon) | (data_s.longitude>mask_Mlon) |\
              (data_s.latitude<mask_mlat) | (data_s.latitude>mask_Mlat),drop=False)

data_s = data_s.drop('adt')

data_s = xr.merge([tmp_adt,data_s])

plt.pcolormesh(data_s.adt.mean(dim='time').values)



###

data_s = data_s.drop(['vgosa','ugosa','sla','err','lon_bnds','lat_bnds','crs'])


data_ADT_a = data_s - data_s.mean(dim='time')


# # stack lat and lon into a single dimension called allpoints
# stacked_adt = data_ADT_a.adt.stack(allpoints=['latitude','longitude'])
# stacked_ugos = data_ADT_a.ugos.stack(allpoints=['latitude','longitude'])
# stacked_vgos = data_ADT_a.vgos.stack(allpoints=['latitude','longitude'])


# # apply the function over allpoints to calculate the trend at each point
# trend_adt = stacked_adt.groupby('allpoints').apply(linear_trend)
# trend_ugos = stacked_ugos.groupby('allpoints').apply(linear_trend)
# trend_vgos = stacked_vgos.groupby('allpoints').apply(linear_trend)


# # unstack back to lat lon coordinates
# trend_unstacked_adt = trend_adt.unstack('allpoints')
# trend_unstacked_ugos = trend_ugos.unstack('allpoints')
# trend_unstacked_vgos = trend_vgos.unstack('allpoints')


# # Detrend
# data_adt_a_dt = data_ADT_a.adt - trend_unstacked_adt
# data_ugos_a_dt = data_ADT_a.ugos - trend_unstacked_ugos
# data_vgos_a_dt = data_ADT_a.vgos - trend_unstacked_vgos

# Merge data
# data_concat = xr.merge([data_adt_a_dt,data_ugos_a_dt,data_vgos_a_dt])

data_concat = data_ADT_a
data_concat = data_concat.fillna(-999)


WD = 2*12
data_concat_2Y = data_concat.rolling(time=WD,center=True).mean().dropna("time")
data_concat_2Y = data_concat_2Y.where(data_concat_2Y!=-999,drop=False) 

# figdata
lat31 = data_concat_2Y.latitude
lon31 = data_concat_2Y.longitude

figdata31 = data_concat_2Y.adt.values
figdata32 = data_concat_2Y.ugos
figdata33 = data_concat_2Y.vgos

lon_m31, lat_m31 = np.meshgrid(lon31,lat31)

r_x, r_y, r_data1, r_data2 = r_vector4cube(lon31,lat31,figdata32,figdata33,[3,3])

lon_m32, lat_m32 = np.meshgrid(r_x,r_y)

figdata32 = r_data1
figdata33 = r_data2


r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300])

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')


# ----figure -----
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

w_path_sig = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_adt/1/'
n = 12
while n < 312:  
    fig, ax = plt.subplots(figsize=(18,5),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
                llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
    # lon_m, lat_m = np.meshgrid(lon_00,lat_00)
    # x, y = m(lon, lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    plt.title('a) Date : '+Sig_set.dates[n] + ' (ADT & Geo UVa 2Y filtered)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs1 = m.contour(lon_m11,lat_m11,np.flipud(figdata111[n,:,:]),colors='grey',linewidths=2.5,levels=10)
    # plt.clim(-3.3,3.3)
    # plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

    cs2 = m.pcolormesh(lon_m31,lat_m31,figdata31[n-12,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    plt.clim(-1.5,1.5) # plt.clim(-max_figdata02,max_figdata02)
    
    q = m.quiver(lon_m32,lat_m32,figdata32[n-12,:,:],figdata33[n-12,:,:],
         scale=3.5,headwidth=7.5,headaxislength=10,headlength=13,color='k',
         minlength=1,edgecolor='y',minshaft=1.3,alpha=.7)
    # plt.axis('equal')
    # Unit vector
    p = plt.quiverkey(q,115.,27,.3,"0.3 m/s",coordinates='data',color='r',
                      labelpos='S',alpha=1,labelcolor='w',fontproperties={'size':16},
                      labelsep=0.13)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    # h = plt.colorbar(label='',cax=cax);
    h = plt.colorbar(label='10 [factor]',cax=cax);
    # plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(w_path_sig+'ADTa_GeoUVa_'+Sig_set.dates[n])
    plt.show()
    n+=1


















