#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 13:55:13 2021

@author: caelus
"""


PKG_path = '/home/caelus/dock_1/Working_hub/LGnDC_dep/python_cent/MantaPKG/'

import sys 
sys.path.append(PKG_path)

from Manta_Signals.procc_index import sig_pro, linearRegress4Cube,running_corr
from Manta_Signals.utility import nc2npy
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/EKE/qiu_wang/'

w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/fig2/'


Sig_set,_,Annual_mean = sig_pro(r_path1,['1993-01-01',324,300],Standard=True)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

# Sv check (Yan 2016)

plt.figure(1)
plt.plot(Annual_mean.index,Annual_mean.ADT_index,color='k')
plt.scatter(Annual_mean.index,Annual_mean.ADT_index,color='darkred')
plt.xticks(fontsize=12, alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='y', alpha=.3)
plt.grid(axis='x', alpha=.3)


plt.figure(2)
plt.plot(Sig_set.index,Sig_set.ADT_index_2Y_Rm,color='k')
plt.xticks(fontsize=12, alpha=.7)
plt.yticks(fontsize=12, alpha=.7)
plt.grid(axis='y', alpha=.3)
plt.grid(axis='x', alpha=.3)



Annual_mean.keys()


# Compare sigs


# Compare sigs
plt.figure(1,figsize=(16,7),dpi=80)
ax = plt.gca()

plt.plot(Annual_mean.index,Annual_mean.ADT_index, label='ADT index (annual mean)',color='darkred',linewidth=2.5,zorder=5)
plt.scatter(Annual_mean.index,Annual_mean.ADT_index,color='darkred',zorder=6)

# plt.plot(Annual_mean.index,Annual_mean.EKE_qiu_10_30_120_250, label='EKE qiu (annual mean)',color='darkblue',linewidth=2.5,zorder=3)
# plt.scatter(Annual_mean.index,Annual_mean.EKE_qiu_10_30_120_250,color='darkblue',zorder=4)

# plt.plot(Annual_mean.index,Annual_mean.PDO, label='PDO (annual mean)',color=[.65,.65,.65],linewidth=2.5,zorder=1)
# plt.scatter(Annual_mean.index,Annual_mean.PDO,color=[.65,.65,.65],zorder=2)

plt.bar(Annual_mean.index,Annual_mean.MEIv2, label='MEIv2 (annual mean)',color=[.9,.9,.9],linewidth=2.5,zorder=0,alpha=.7)

# plt.plot(Annual_mean.index,-Annual_mean.WP, label='-WP (annual mean)',color='darkgreen',linewidth=2.5,zorder=3)
# plt.scatter(Annual_mean.index,-Annual_mean.WP,color='darkgreen',zorder=2)


xtick_location = Annual_mean.index.tolist()[::2]
xtick_labels = Annual_mean.index.tolist()[::2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=30, fontsize=16, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=20, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.xticks(fontsize=20, alpha=.7,fontweight='bold')
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower right',fontsize=12)
plt.grid(axis='x', alpha=.3)
# if savefig:
#     plt.savefig(w_path1+'Index_nega_WP_annual',dpi=150)
plt.show()








# Running corr

sig1,sig2 = Annual_mean.ADT_index, Annual_mean.EKE_qiu_10_30_120_250

Corr_vector = []
factor = int(1) #int(WY*12/2)
    
for i in range(int(factor),len(sig1)-factor):
    # print(i)
    Corr_vector.append(np.corrcoef(sig1[i-factor:i+factor],sig2[i-factor:i+factor])[0,1])
    # print(Corr_vector[i-factor],sig1.date[i])


plt.plot(Corr_vector)



# Compare sigs
plt.figure(2,figsize=(16,10),dpi=80)
ax = plt.gca()

plt.plot(Annual_mean.index,Annual_mean.ADT_index, label='ADT index (annual mean)',color='darkred',linewidth=2.5,zorder=5)
plt.scatter(Annual_mean.index,Annual_mean.ADT_index,color='darkred',zorder=6)

plt.plot(Annual_mean.index,Annual_mean.EKE_qiu_10_30_120_250, label='EKE qiu (annual mean)',color='darkblue',linewidth=2.5,zorder=3)
plt.scatter(Annual_mean.index,Annual_mean.EKE_qiu_10_30_120_250,color='darkblue',zorder=4)

plt.plot(Annual_mean.index[factor:-factor],Corr_vector, label='Time-dependent Corr ('+str(2)+'Y)',color=[.77,.77,.77],
          zorder=0,linewidth=2.5)

plt.bar(Annual_mean.index,Annual_mean.MEIv2, label='MEIv2 (annual mean)',color=[.9,.9,.9],linewidth=2.5,zorder=0,alpha=.7)

# plt.plot(Annual_mean.index,Annual_mean.WP, label='WP (2Y Runing mean)',color='g',linewidth=2.5,zorder=3)
xtick_location = Annual_mean.index.tolist()[::2]
xtick_labels = Annual_mean.index.tolist()[::2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=30, fontsize=16, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=20, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.xticks(fontsize=20, alpha=.7,fontweight='bold')
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower right',fontsize=14)
plt.grid(axis='x', alpha=.3)
# if savefig:
    # plt.savefig(w_path1+'rcorr_annual',dpi=150)
plt.show()














