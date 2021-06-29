
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 21:03:59 2021

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


r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/CDS/'
# r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/'
# r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/'


data = xr.open_mfdataset(r_path1+'*.nc', parallel=True)
ADT= data.adt

ADT = ADT.loc[dict(latitude=slice(23,24),longitude=slice(120.5,180))]


data_a = ADT - ADT.mean(dim='time')

data_a = data_a.fillna(-999)
WD = 2*12
data_a_2Y = data_a.rolling(time=WD,center=True).mean().dropna("time")
data_a_2Y = data_a_2Y.where(data_a_2Y!=-999,drop=False)

data_a_2Y = data_a_2Y.mean(dim='latitude')

# 6MOnth mean
# WD = 6
data_a_6M = data_a.rolling(time=6,center=True).mean().dropna("time")
data_a_6M = data_a_6M.where(data_a_6M!=-999,drop=False)
data_a_6M = data_a_6M.mean(dim='latitude')

# 3Y mean
data_a_3Y = data_a.rolling(time=12*3,center=True).mean().dropna("time")
data_a_3Y = data_a_3Y.where(data_a_3Y!=-999,drop=False)
data_a_3Y = data_a_3Y.mean(dim='latitude')


data_a_6M = data_a_6M.loc[dict(time=slice('1994-01','2018-12'))]
data_a_2Y = data_a_2Y.loc[dict(time=slice('1994-01','2018-12'))]
data_a_3Y = data_a_3Y.loc[dict(time=slice('1994-01','2018-12'))]


t10 = data_a_2Y.time.values
lon10 = data_a_2Y.longitude.values 
figdata10 = data_a_2Y.values

lon_m10, t_m10 = np.meshgrid(lon10,t10)

# --------

t11 = data_a_6M.time.values
lon11 = data_a_6M.longitude.values 
figdata11 = data_a_6M.values

lon_m11, t_m11 = np.meshgrid(lon11,t11)

# --------------------
t12 = data_a_3Y.time.values
lon12 = data_a_3Y.longitude.values 
figdata12 = data_a_3Y.values

lon_m12, t_m12 = np.meshgrid(lon12,t12)

# t12 = data_a.time.values
# lon11 = data_a.longitude.values 
# figdata11 = data_a.values

# lon_m11, t_m11 = np.meshgrid(lon11,t11)




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

# =============================================================================
# Label
# =============================================================================

t_label= pd.DataFrame({'date':pd.date_range('1994-01-01', periods = 300,freq = 1 * '1m').strftime('%Y')})
lon_label = []
for i in lon10:
    lon_label.append(str(i)[:3]+'°E')


# --------figure 3 Y
w_path01 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/figs/'

# fig, ax = plt.subplots(figsize=(8,15),linewidth=1)
# plt.pcolormesh(lon_m12, t_m12,figdata12*10,
#                cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# # plt.colorbar()
# plt.clim(-1,1)
# plt.axvline(x=lon10[19],color='k',linewidth=3,linestyle='--',alpha=.6)

# ytick_location = t10[::12]
# ytick_labels = t_label.date.tolist()[::12]
# plt.yticks(ticks=ytick_location, labels=ytick_labels, rotation=0, fontsize=18, alpha=.7)
# plt.ylabel('$\it{Years}$',fontsize=20,fontweight='light',alpha=.7)
# plt.title('b) 3Y filtered (23.5$\degree$N)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='bold')

# xtick_location = lon10[20::40]
# xtick_labels = lon_label[20::40]
# plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=18, alpha=.7)
# plt.xlabel('$\it{longitude }$',fontsize=20,fontweight='light',alpha=.7)
# plt.grid(axis='y', alpha=.3,linestyle='-.',color='k')
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="2.5%", pad=0.1)
# cax.tick_params(labelsize=15)
# cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic','alpha':.7})
# #label 
# h = plt.colorbar(label='$\mathit{10^{-1}[m]}$',cax=cax);
# # plt.savefig(w_path01+'EKEa_2YM',bbox_inches='tight')
# plt.tight_layout()
# plt.show()

# ---------figure 2Y
fig, ax = plt.subplots(figsize=(8,15),linewidth=1)
plt.pcolormesh(lon_m10, t_m10,figdata10*10,
               cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# plt.colorbar()
plt.clim(-1,1)
plt.axvline(x=lon10[19],color='k',linewidth=3,linestyle='--',alpha=.6)

ytick_location = t10[::12]
ytick_labels = t_label.date.tolist()[::12]
plt.yticks(ticks=ytick_location, labels=ytick_labels, rotation=0, fontsize=18, alpha=.7)
plt.ylabel('$\it{Years}$',fontsize=20,fontweight='light',alpha=.7)
plt.title('c) 2Y filtered (23.5$\degree$N)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='bold')

xtick_location = lon10[20::40]
xtick_labels = lon_label[20::40]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=18, alpha=.7)
plt.xlabel('$\it{longitude }$',fontsize=20,fontweight='light',alpha=.7)
plt.grid(axis='y', alpha=.3,linestyle='-.',color='k')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic','alpha':.7})
#label 
h = plt.colorbar(label='$\mathit{10^{-1}[m]}$',cax=cax);
# plt.savefig(w_path01+'EKEa_2YM',bbox_inches='tight')
plt.tight_layout()
plt.show()

# -----------------
# t_label2= pd.DataFrame({'date':pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y')})
# lon_label2 = []
# for i in lon11:
#     lon_label2.append(str(i)[:3]+'°E')
    
    
fig, ax = plt.subplots(figsize=(8,15),linewidth=1)
plt.pcolormesh(lon_m11, t_m11,figdata11*10,
               cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# plt.colorbar()
plt.clim(-1,1)
plt.axvline(x=lon10[19],color='k',linewidth=3,linestyle='--',alpha=.6)

# ytick_location = t11[::12]
# ytick_labels = t_label.date.tolist()[::12]
plt.yticks(ticks=ytick_location, labels=ytick_labels, rotation=0, fontsize=18, alpha=.7)
plt.ylabel('$\it{Years}$',fontsize=20,fontweight='light',alpha=.7)
plt.title('b) 6M filtered (23.5$\degree$N)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='bold')

xtick_location = lon11[20::40]
xtick_labels = lon_label[20::40]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=18, alpha=.7)
plt.xlabel('$\it{longitude }$',fontsize=20,fontweight='light',alpha=.7)
plt.grid(axis='y', alpha=.3,linestyle='-.',color='k')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic','alpha':.7})
#label 
h = plt.colorbar(label='$\mathit{10^{-1}[m]}$',cax=cax);
# plt.savefig(w_path01+'EKEa_2YM',bbox_inches='tight')
plt.tight_layout()
plt.show()


# =============================================================================
# =============================================================================
# # Bo qiu EKE regression / adt
# =============================================================================
# =============================================================================
r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300])

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')

Sig_set = Sig_set.loc[slice('1994','2018-12')]

tmp_MEIP = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm >= 0)
tmp_MEIN = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm < 0)

plt.figure(1,figsize=(8,16),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.title('a) Sv anomaly (2Y Rm)', fontproperties='',loc='center',pad=20,  fontsize=28,fontweight='bold')
plt.plot(Sig_set.KVTe_index_2Y_Rm, t11,label='KVTe (Yan & Sun 2015)',color='k',linewidth=3,zorder=10)
plt.plot(Sig_set.EKE_qiu_10_30_120_250_2Y_Rm, t11,label='EKE (Qiu 2013)',color='red',linewidth=3,zorder=9)
plt.plot(Sig_set.PDO_2Y_Rm, t11,label='PDO',color='blue',linewidth=3,zorder=9)
plt.ylim(t11[0],t11[-2])
plt.xlim(-3,3)

# plt.fill_between(tmp_MEIP.values,t11,  color="lightpink",
#                   alpha=0.5, label='El-nino',zorder=0)
# plt.fill_between(tmp_MEIN,t11,  color="skyblue",
#                   alpha=0.5, label='La nina',zorder=1)

plt.axvline(x=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)
# plt.axvline(x=Sig_set.dates[n],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=0)

# Decoration
# plt.ylim(50,750)
# xtick_location = Sig_set.dates.tolist()[::12*2]
# xtick_labels = Sig_set.dates.tolist()[::12*2]
plt.yticks(ticks=ytick_location, labels=ytick_labels, rotation=0, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=20, alpha=.7)
plt.xticks(fontsize=20, alpha=.7)

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











