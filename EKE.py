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


r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/'
r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/'


data = xr.open_mfdataset(r_path2+'*.nc', parallel=True)
data['EKE'] = (data.ugosa**2 + data.vgosa**2)/2

data_a = data - data.mean(dim='time')

WD = 2*12
data_a_2Y = data_a.rolling(time=WD,center=True).mean().dropna("time")


EKE_2Y_latM = data_a_2Y.EKE.mean(dim='y')


t10 = EKE_2Y_latM.time.values
lon10 = EKE_2Y_latM.lon.values 
figdata10 = EKE_2Y_latM.values

lon_m10, t_m10 = np.meshgrid(lon10,t10)
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



# --------figure
w_path01 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/figs/'

fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
plt.pcolormesh(lon_m10, t_m10,figdata10*10**4,
               cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# plt.colorbar()
plt.clim(-100,100)

ytick_location = t10[6::12*2]
ytick_labels = t_label.date.tolist()[6::12*2]
plt.yticks(ticks=ytick_location, labels=ytick_labels, rotation=0, fontsize=18, alpha=.7)
plt.ylabel('$\it{Years}$',fontsize=20,fontweight='light',alpha=.7)
plt.title(' EKE anomaly ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')

xtick_location = lon10[::40]
xtick_labels = lon_label[::40]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=18, alpha=.7)
plt.xlabel('$\it{longitude }$',fontsize=20,fontweight='light',alpha=.7)
plt.grid(axis='y', alpha=.3,linestyle='-.',color='k')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic','alpha':.7})
#label 
h = plt.colorbar(label='$\mathit{[(cm/s)^{2}]}$',cax=cax);
plt.savefig(w_path01+'EKEa_2YM',bbox_inches='tight')
plt.show()


# =============================================================================
# =============================================================================
# # Bo qiu EKE regression / adt
# =============================================================================
# =============================================================================

Sig_set,Corr_map,Annual_mean = sig_pro(r_path1,['1993-01-01',324,300],Standard=True)

data_adt = xr.open_mfdataset(r_path3+'adt_0_60_112_260_M.nc', parallel=True)

adt = data_adt.adt.values

Sig_set.keys()

Sig = Sig_set.ADT_index_2Y_Rm.dropna()[:-1]


Coef_adt1, _ = linearRegress4Cube(Sig,adt[12:-12,:,:],['1994-01','2018-12'])
Coef_adt2, _ = linearRegress4Cube(Sig,adt[12:-12,:,:],['1994-01','2005-10'])
Coef_adt3, _ = linearRegress4Cube(Sig,adt[12:-12,:,:],['2005-11','2011-01'])
Coef_adt4, _ = linearRegress4Cube(Sig,adt[12:-12,:,:],['2005-11','2013-03'])
Coef_adt5, _ = linearRegress4Cube(Sig,adt[12:-12,:,:],['2013-04','2018-12'])

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/Coef_YS_adt_199401_201812',Coef_adt1)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/Coef_YS_adt_199401_200510',Coef_adt2)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/Coef_YS_adt_200511_201101',Coef_adt3)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/Coef_YS_adt_200511_201303',Coef_adt4)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/Coef_YS_adt_201304_201812',Coef_adt5)



lat20 = data_adt.lat.values
lon20 = data_adt.lon.values 
lon_m20, lat_m20 = np.meshgrid(lon20,lat20)


# -----------------------figure----------------------------


w_path21 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/figs/'
minlat,maxlat,minlon,maxlon = 0,60,112,260


fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
            llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
plt.title(' Adt reg (EKE) 2013-04 ~ 2018-12  ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)
# cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')

cs2 = m.pcolormesh(lon_m20,lat_m20,Coef_adt2*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-2,2)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='10 [factor]',cax=cax);
# plt.savefig(w_path21+'adt_regressed_YS_201304_201812',bbox_inches='tight')
plt.tight_layout()
plt.show()














