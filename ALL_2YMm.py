#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 21:00:08 2021

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
# WSC & Press
# =============================================================================

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/ERA5/' 


Time = ['1993-01','2019-12'] 
minlon,maxlon = 112,260
minlat,maxlat = -10,70,
    

Sample_Data1 = xr.open_dataset(r_path1+'ERA5_landSea_mask.nc')
Sample_Data2 = xr.open_dataset(r_path1+'ERA5_single_level.nc')

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

data_WSC = data_WSC.drop(['sp','tp','u10','v10','lsm'])

data_WSC_a = data_WSC - data_WSC.mean(dim='time')

data_WSC_a = data_WSC_a.fillna(-999)

WD = 2*12
data_WSC_a_2Y = data_WSC_a.rolling(time=WD,center=True).mean().dropna("time")

data_WSC_a_2Y = data_WSC_a_2Y.where(data_WSC_a_2Y!=-999,drop=False) 


figdata11 = data_WSC_a_2Y.msl.values/100 # Pa --> hpa
figdata12 = data_WSC_a_2Y.curlZ.values
lon_10 = data_WSC_a_2Y.longitude.values  
lat_10 = np.flipud(data_WSC_a_2Y.latitude.values)
lon_m10, lat_m10 = np.meshgrid(lon_10,lat_10)

# ------------ test fig WSC ----------------

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
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
cs1 = m.contour(lon_m10,lat_m10,np.flipud(figdata11[70,:,:]),colors='grey',linewidths=2.5,levels=10)
plt.clim(-3.3,3.3)
plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(figdata12[70,:,:])*10**7,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-.3,.3)
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
# EKE
# =============================================================================

r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/'

data_EKE = xr.open_mfdataset(r_path2+'*.nc', parallel=True)
data_EKE['EKE'] = (data_EKE.ugosa**2 + data_EKE.vgosa**2)/2

data_EKE_a = data_EKE - data_EKE.mean(dim='time')

data_EKE_a = data_EKE_a.drop(['vgosa','ugosa'])
data_EKE_a = data_EKE_a.fillna(-999)

WD = 2*12
data_EKE_a_2Y = data_EKE_a.rolling(time=WD,center=True).mean().dropna("time")
data_EKE_a_2Y = data_EKE_a_2Y.where(data_EKE_a_2Y!=-999,drop=False) 



lat20 = data_EKE_a_2Y.lat.values
lon20 = data_EKE_a_2Y.lon.values 
figdata20 = data_EKE_a_2Y.EKE.values

lon_m20, lat_m20 = np.meshgrid(lon20,lat20)

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
# plt.title(' Adt reg (EKE) 2013-04 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m20,lat_m20,figdata20[100,:,:]*100,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
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
# ADT
# =============================================================================

r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_adt/'

data_ADT = xr.open_mfdataset(r_path3+'*.nc', parallel=True)

data_ADT_a = data_ADT - data_ADT.mean(dim='time')

data_ADT_a = data_ADT_a.fillna(-999)

WD = 2*12
data_ADT_a_2Y = data_ADT_a.rolling(time=WD,center=True).mean().dropna("time")
data_ADT_a_2Y = data_ADT_a_2Y.where(data_ADT_a_2Y!=-999,drop=False) 


lat30 = data_ADT_a_2Y.lat.values
lon30 = data_ADT_a_2Y.lon.values 
figdata30 = data_ADT_a_2Y.adt.values

lon_m30, lat_m30 = np.meshgrid(lon30,lat30)



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
# plt.title(' Adt reg (EKE) 2013-04 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m30,lat_m30,figdata30[100,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
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






# =============================================================================
# Signal
# =============================================================================
r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300],Standard=True)

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')


# ------------ test fig ----------------


plt.figure(1,figsize=(16,7),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot

plt.plot(Sig_set.dates,Sig_set.ADT_index_2Y_Rm, label='Kuroshio_Rm (2Y Runing mean)',color='k',linewidth=2.5)
plt.plot(Sig_set.dates,Sig_set.PDO_2Y_Rm, label='EKE_wang_pc1_2Y_Rm (2Y Runing mean)',color='darkred',linewidth=2.5)
plt.plot(Sig_set.dates,Sig_set.WP_2Y_Rm, label='EKE_wang_pc1_2Y_Rm (2Y Runing mean)',color='darkblue',linewidth=2.5)
# plt.bar(Annual_mean.index,Annual_mean.MEIv2, label='MEIv2',color=[.9,.9,.9],linewidth=2.5,zorder=0,alpha=.7)

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3)
plt.axvline(x=Sig_set.dates[20],color='k',linewidth=3,linestyle='--',alpha=.9)

# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.dates.tolist()[::12*2]
xtick_labels = Sig_set.dates.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=45, fontsize=22, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=22, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=.6)
plt.show()




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

tmp_MEIP = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm >= 0)
tmp_MEIN = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm < 0)

w_path_sig = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/Figures/SUM/'
n = 12
while n < 312:  
    plt.figure(1,figsize=(16,6),dpi=80)
    ax = plt.gca()
    # plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
    #           fontweight='bold',fontsize='48')
    
    # plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
    # Draw Plot
    plt.title('a) Date : '+Sig_set.dates[n] + ' (2Y running mean)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
    plt.plot(Sig_set.dates,Sig_set.ADT_index_2Y_Rm, label='YS index (Yan & Sun 2015)',color='k',linewidth=3,zorder=10)
    plt.plot(Sig_set.dates,Sig_set.EKE_qiu_10_30_120_250_2Y_Rm, label='EKE (Qiu 2013)',color='darkred',linewidth=3,zorder=9)
    plt.plot(Sig_set.dates,Sig_set.PDO_2Y_Rm, label='PDO ',color='darkblue',linewidth=2.5,zorder=8)
    plt.plot(Sig_set.dates,Sig_set.WP_2Y_Rm, label='WP ',color='green',linewidth=2.5,zorder=7)
    
    # Scatter
    
    plt.scatter(Sig_set.dates[n],Sig_set.ADT_index_2Y_Rm[n],
                color='k',marker='o',s=200,zorder=20)
    plt.scatter(Sig_set.dates[n],Sig_set.EKE_qiu_10_30_120_250_2Y_Rm[n],
                color='darkred',marker='o',s=200,zorder=19)
    plt.scatter(Sig_set.dates[n],Sig_set.PDO_2Y_Rm[n],
                color='darkblue',marker='o',s=200,zorder=18)
    plt.scatter(Sig_set.dates[n],Sig_set.WP_2Y_Rm[n],
                color='green',marker='o',s=200,zorder=17)
    
    
    plt.fill_between(Sig_set.dates, tmp_MEIP, color="lightpink",
                      alpha=0.5, label='El-nino',zorder=0)
    plt.fill_between(Sig_set.dates, tmp_MEIN, color="skyblue",
                      alpha=0.5, label='La nina',zorder=1)
    
    plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)
    plt.axvline(x=Sig_set.dates[n],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=0)
    
    # Decoration
    # plt.ylim(50,750)
    xtick_location = Sig_set.dates.tolist()[::12*2]
    xtick_labels = Sig_set.dates.tolist()[::12*2]
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
    # plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.yticks(fontsize=18, alpha=.7)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.legend(loc='lower right',fontsize=10)
    plt.grid(axis='y', alpha=.6)
    plt.tight_layout()
    plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
    plt.show()

# ------------ fig WSC ----------------

    
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
    plt.title('b) Date : '+Sig_set.dates[n] + ' (WSCa & Pressa 2Y filtered)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs = m.pcolormesh(lon_m,lat_m,data)
    # cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    cs1 = m.contour(lon_m10,lat_m10,np.flipud(figdata11[n-12,:,:]),colors='grey',linewidths=2.5,levels=10)
    plt.clim(-3.3,3.3)
    plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')
    
    cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(figdata12[n-12,:,:])*10**7,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.clim(-max_figdata02,max_figdata02)
    plt.clim(-.3,.3)
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
    plt.savefig(w_path_sig+'/WSCa/WCSa_'+Sig_set.dates[n])
    plt.show()

# ------------ fig ADT ----------------
    
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
    plt.title('c) Date : '+Sig_set.dates[n] + ' (ADTa 2Y filtered)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
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
    plt.savefig(w_path_sig+'/ADTa/ADTa_'+Sig_set.dates[n])
    plt.tight_layout()
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
    plt.title('d) Date : '+Sig_set.dates[n] + ' (EKEa 2Y filtered)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs = m.pcolormesh(lon_m,lat_m,data)
    # cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    
    cs2 = m.pcolormesh(lon_m20,lat_m20,figdata20[n-12,:,:]*100,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.clim(-max_figdata02,max_figdata02)
    plt.clim(-2,2)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='100 [factor]',cax=cax);
    plt.savefig(w_path_sig+'/EKEa/EKEa_'+Sig_set.dates[n])
    plt.tight_layout()
    plt.show()
    
    n+=1





























