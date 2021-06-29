#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:55:40 2021

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


r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/CDS/'

Data = xr.open_dataset(r_path1+'Detrended_CDS_monthly_199301_201912.nc')

Time = ['1993-01','2019-12'] 
minlon,maxlon = 123.4,123.7
minlat,maxlat = 23.9,24.2


lon = Data.longitude.values
lat = Data.latitude.values


lat_co = np.where((lat >= minlat) & (lat <= maxlat))[0]
lon_co = np.where((lon >= minlon) & (lon <= maxlon))[0]

data_s = Data.loc[dict(latitude=lat[lat_co],longitude=lon[lon_co])]

# -----------------

minlon2,maxlon2 = 123.15,123.4
minlat2,maxlat2 = 27.2,27.4


lon2 = Data.longitude.values
lat2 = Data.latitude.values

lat_co2 = np.where((lat2 >= minlat2) & (lat2 <= maxlat2))[0]
lon_co2 = np.where((lon2 >= minlon2) & (lon2 <= maxlon2))[0]

data_s2 = Data.loc[dict(latitude=lat2[lat_co2],longitude=lon2[lon_co2])]


YS_ori = data_s.adt.values - data_s2.adt.values

YS = pd.DataFrame({'YS':YS_ori.reshape(-1)},index=data_s.time.values)


YS_2Y = YS.rolling(window=12*2,center=True).mean()
YS_2Y['ori'] = YS_ori.reshape(-1)

YS_2Y.plot()

plt.plot(YS_2Y)

A = YS_2Y['ori'].values

KVTe_index = A*.31+6.55

np.save('/home/caelus/tmp/KVTe',KVTe_index)

# =============================================================================
# 
# =============================================================================
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
plt.plot(Sig_set.dates,YS.zeta.values.reshape(-1), label='YS index (zeta)',color='k',linewidth=3,zorder=10)
plt.plot(Sig_set.dates[12:-12],YS_2Y.zeta.values.reshape(-1)[:-1], label='YS index (zeta 2Y running mean)',color='darkred',linewidth=3,zorder=9)
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
plt.legend(loc='lower right',fontsize=16)
plt.grid(axis='y', alpha=.6)
plt.tight_layout()
# plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
plt.show()















