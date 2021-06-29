#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 18:49:07 2021

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

r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300])
Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')

minlon,maxlon = 112,180
minlat,maxlat = 5,30

KVTe = Sig_set.KVTe_index_2Y_Rm

ADT_t = xr.open_dataset('/home/caelus/dock_1/Working_hub/DATA_dep/CDS/T_CDS_monthly_199301_201912.nc',decode_times=True)

ADT_t = ADT_t.drop(['crs','lat_bnds','lon_bnds','err','sla','ugosa','vgosa'])

ADT_t = ADT_t.loc[dict(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),nv= 1 )]

# ------Masking (Extension) --------

mask_mlon,mask_Mlon = 130,190
mask_mlat,mask_Mlat = 30, 40

tmp_data = ADT_t.adt 

ADT_t =ADT_t.where( (ADT_t.longitude<mask_mlon) | (ADT_t.longitude>mask_Mlon) |\
              (ADT_t.latitude<mask_mlat) | (ADT_t.latitude>mask_Mlat),drop=False)

ADT_t = ADT_t.drop('adt')

ADT_t = xr.merge([tmp_data,ADT_t])
# --------------


ADT_t['EKE'] = (ADT_t.ugos*2 + ADT_t.vgos*2)/2

ADT_t = ADT_t - ADT_t.mean(dim='time')



Time_p1 = ['1994-12','1999-01']
Time_p2 = ['2003-01','2005-03']
Time_p3 = ['2006-12','2009-07']

P1_data = ADT_t.loc[dict(time=slice(Time_p1[0],Time_p1[1]))]
P2_data = ADT_t.loc[dict(time=slice(Time_p2[0],Time_p2[1]))]
P3_data = ADT_t.loc[dict(time=slice(Time_p3[0],Time_p3[1]))]

P1 = P1_data.mean(dim='time')
P2 = P2_data.mean(dim='time')
P3 = P3_data.mean(dim='time')


# ------------- 
def r_vector4cube(x,y,data1,data2,factor):
    xx = x.values.shape[0]
    yy = y.values.shape[0]
    a = np.arange(0,xx,factor[0])
    b = np.arange(0,yy,factor[1])
    
    r_x, r_y = x[a], y[b]
    
    r_data1 = data1.where( (data1.longitude==r_x) & (data1.latitude==r_y), drop=True )
    r_data2 = data2.where( (data2.longitude==r_x) & (data2.latitude==r_y), drop=True )    
    
    return r_x, r_y, r_data1, r_data2 


figdata11 = P2.adt.values
lon11 = P1.adt.longitude
lat11 = P1.adt.latitude

r_x12, r_y12, r_data121,r_data122 = r_vector4cube(P1.ugos.longitude,P1.ugos.latitude,
                                                  P2.ugos,P2.vgos,[4,4])

figdata121 = r_data121
figdata122 = r_data122

lon12 = r_x12
lat12 = r_y12

figdata13 = P2.EKE.values
# lon13 = P1.adt.longitude
# lat13 = P1.adt.latitude

lon_m11, lat_m11 = np.meshgrid(lon11,lat11)
lon_m12, lat_m12 = np.meshgrid(lon12,lat12)

#-------------

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

# w_path_sig = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_adt/1/'
 
fig, ax = plt.subplots(figsize=(18,5),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# lon_m, lat_m = np.meshgrid(lon_00,lat_00)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=22,fontweight='bold',color='k')
m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=22,fontweight='bold',color='k')
plt.title('b) ADTa UVa Mean [2003-01, 2005-03] ', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs1 = m.contour(lon_m11,lat_m11,np.flipud(figdata111[n,:,:]),colors='grey',linewidths=2.5,levels=10)
# plt.clim(-3.3,3.3)
# plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m11,lat_m11,figdata11*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-1.5,1.5) # plt.clim(-max_figdata02,max_figdata02)

q = m.quiver(lon_m12,lat_m12,figdata121,figdata122,
      scale=2.5,headwidth=7.5,headaxislength=10,headlength=13,color='k',
      minlength=1,edgecolor='y',minshaft=1.3,alpha=.7)
# plt.axis('equal')
# Unit vector
p = plt.quiverkey(q,115.,29,.1,"0.1 m/s",coordinates='data',color='r',
                  labelpos='S',alpha=1,labelcolor='w',fontproperties={'size':16},
                  labelsep=0.13)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
# h = plt.colorbar(label='',cax=cax);
h = plt.colorbar(label='$10^{-1} [m]$',cax=cax);
# plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
plt.tight_layout()
# plt.savefig(w_path_sig+'ADTa_GeoUVa_'+Sig_set.dates[n])
plt.show()
# n+=1




figdata11 = P2.adt.values

# r_x12, r_y12, r_data121,r_data122 = r_vector4cube(P1.ugos.longitude,P1.ugos.latitude,P3.ugos,P3.vgos,[3,3])

# figdata121 = r_data121
# figdata122 = r_data122

lon12 = r_x12
lat12 = r_y12

figdata13 = P2.EKE.values

# ----------EKE -------------

fig, ax = plt.subplots(figsize=(18,5),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# lon_m, lat_m = np.meshgrid(lon_00,lat_00)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=22,fontweight='bold',color='k')
m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=22,fontweight='bold',color='k')
plt.title('[Positive] EKEa Mean [2003-01, 2005-03] ', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs1 = m.contour(lon_m11,lat_m11,figdata11*100,colors='k',linewidths=2.5,levels=10,alpha=.45)
# plt.clim(-300.3,300.3)
# plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m11,lat_m11,figdata13*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-1.5,1.5) # plt.clim(-max_figdata02,max_figdata02)
m.plot([120,180,180,120,120],[18,18,28,28,18],color='k',linestyle='--',linewidth=4,alpha=.8)

# q = m.quiver(lon_m12,lat_m12,figdata121,figdata122,
#       scale=2.5,headwidth=7.5,headaxislength=10,headlength=13,color='k',
#       minlength=1,edgecolor='y',minshaft=1.3,alpha=.7)
# # plt.axis('equal')
# # Unit vector
# p = plt.quiverkey(q,115.,29,.1,"0.1 m/s",coordinates='data',color='r',
#                   labelpos='S',alpha=1,labelcolor='w',fontproperties={'size':16},
#                   labelsep=0.13)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
# h = plt.colorbar(label='',cax=cax);
h = plt.colorbar(label='$\mathit{10^{-1}[(m/s)^{2}]}$',cax=cax);
# plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
plt.tight_layout()
# plt.savefig(w_path_sig+'ADTa_GeoUVa_'+Sig_set.dates[n])
plt.show()
# n+=1










