#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 00:26:14 2021

@author: caelus
"""

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
# Signal
# =============================================================================
r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300],Standard=True)

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')


# ------------ test fig ----------------



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



CP = pd.read_csv('/home/caelus/wormhole/CP/csv/CP.csv',header=None)
EP = pd.read_csv('/home/caelus/wormhole/CP/csv/EP.csv',header=None)

CPEP = pd.DataFrame({'CP':CP.values.reshape(-1),'EP':EP.values.reshape(-1)},
                   index=pd.date_range('1993-01-01', periods = 300,freq = 1 * '1m'))

CPEP_2Y = CPEP.rolling(window=int(12*2),center=True).mean()
CPEP_2Y.columns = ['CP_2Y_Rm','EP_2Y_Rm']
Sig_set = pd.concat([Sig_set,CPEP,CPEP_2Y],axis=1)






plt.figure(1,figsize=(16,6),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
# plt.title('a) Date : '+Sig_set.dates[n] + ' (2Y running mean)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
plt.plot(Sig_set.dates,Sig_set.ADT_index_2Y_Rm, label='YS index (Yan & Sun 2015)',color='k',linewidth=3,zorder=10)
plt.plot(Sig_set.dates,Sig_set.EKE_qiu_10_30_120_250_2Y_Rm, label='EKE (Qiu 2013)',color='darkred',linewidth=3,zorder=9)
plt.plot(Sig_set.dates,Sig_set.PDO_2Y_Rm, label='PDO ',color='darkblue',linewidth=2.5,zorder=8)
plt.bar(Sig_set.dates,Sig_set.CP_2Y_Rm, label='CP ',color='C2',linewidth=2.5,zorder=7)
plt.bar(Sig_set.dates,Sig_set.EP_2Y_Rm, label='EP ',color='C6',linewidth=2.5,zorder=7)

# Scatter

# plt.scatter(Sig_set.dates[n],Sig_set.ADT_index_2Y_Rm[n],
#             color='k',marker='o',s=200,zorder=20)
# plt.scatter(Sig_set.dates[n],Sig_set.EKE_qiu_10_30_120_250_2Y_Rm[n],
#             color='darkred',marker='o',s=200,zorder=19)
# plt.scatter(Sig_set.dates[n],Sig_set.PDO_2Y_Rm[n],
#             color='darkblue',marker='o',s=200,zorder=18)
# plt.scatter(Sig_set.dates[n],Sig_set.WP_2Y_Rm[n],
#             color='green',marker='o',s=200,zorder=17)
plt.fill_between(Sig_set.dates, tmp_MEIP, color="lightpink",
                  alpha=0.5, label='El-nino',zorder=0)
plt.fill_between(Sig_set.dates, tmp_MEIN, color="skyblue",
                  alpha=0.5, label='La nina',zorder=1)

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)
# plt.axvline(x=Sig_set.dates[n],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=0)

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
# plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
plt.show()

























