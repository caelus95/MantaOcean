#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:19:47 2021

@author: caelus
"""


PKG_path = '/home/caelus/dock_1/Working_hub/LGnDC_dep/python_cent/MantaPKG/'

import sys 
sys.path.append(PKG_path)

from Manta_Signals.procc_index import sig_pro, linearRegress4Cube
from Manta_Signals.utility import nc2npy
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


r_path = '/home/caelus/dock_1/Working_hub/ROMS_dep/ROMS/'


EOF = np.load(r_path+'eof_EKE.npy')
PC = np.load(r_path+'pc_EKE.npy')

lon = np.load(r_path+'lon_EKE.npy')
lat = np.load(r_path+'lat_EKE.npy')
expvar = np.load(r_path+'expvar_EKE.npy')

lon_m10,lat_m10 = np.meshgrid(lon,lat)
figdata10 = EOF[0,:,:]

# fig 11111111111111111111111
fig, ax = plt.subplots(figsize=(16,7),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],\
            llcrnrlon=lon[0]-5,urcrnrlon=lon[-1],resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81,5),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
# plt.title('c) Date : '+Sig_set.dates[n] + ' (ADTa 2Y filtered)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

# m.plot([120,180,180,120,120],[18,18,28,28,18],color='k',linestyle='--',linewidth=4,alpha=.8)

cs2 = m.pcolormesh(lon_m10,lat_m10,figdata10*10**(2),cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-1.5,1.5)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='10 [factor]',cax=cax);
# plt.savefig(w_path_sig+'/ADTa/ADTa_'+Sig_set.dates[n])
plt.tight_layout()
plt.show()


date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m')
plt.figure(figsize=(20,5))
plt.plot(date[:],PC[0,:])

A = pd.DataFrame({'PC':PC[1,:]},index=date)
A_2Y = A.rolling(window=int(12*2),center=True).mean()

plt.figure(figsize=(20,5))
plt.plot(date,A_2Y)

# fig 22222222222222222222222222


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300])

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')

plt.figure(1,figsize=(16,6),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
# plt.title('a) Date : '+Sig_set.dates[n] + ' (2Y running mean)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
plt.plot(Sig_set.dates,PC[1,:], label='EKE (Qiu 2013 PC1)',color='k',linewidth=3,zorder=10)
plt.plot(Sig_set.dates,A_2Y, label='EKE (Qiu 2013 PC1 2Y)',color='darkred',linewidth=3,zorder=9)
# plt.plot(Sig_set.dates,Sig_set.KVTe_index_2Y_Rm*10**(-1), label='PDO ',color='darkblue',linewidth=2.5,zorder=8)

xtick_location = Sig_set.dates[12:-12].tolist()[::12*2]
xtick_labels = Sig_set.dates[12:-12].tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower left',fontsize=16)
plt.grid(axis='y', alpha=.6)
plt.tight_layout()
# plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
plt.show()










