#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 00:16:29 2021

@author: caelus
"""


import numpy as np
# import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os 


# r_path10 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/EKE/qiu_wang/EOF_figs/wang_225_25_123_124/pc1_300.npy'
# r_path11 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/EKE/qiu_wang/EOF_figs/wang_225_25_123_124/ceemd_EKE_spatial_mean.npy'
savefig = False

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set_list = np.sort([file for file in os.listdir(r_path) if file.endswith(".npy")])

Sig_sets = pd.DataFrame({},index = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m'))

for i in Sig_set_list:
    tmp = np.load(r_path+i).reshape(-1)
    columns_name = i.split('.')[0]
    if len(tmp) == 324 :
        Sig_sets[columns_name] = tmp
    elif len(tmp) == 300 :
        tmp_Sig_set = pd.DataFrame({columns_name:tmp},index = pd.date_range('1994-01-01', periods = 300,freq = 1 * '1m'))
        Sig_sets = pd.concat([Sig_sets,tmp_Sig_set],axis=1)
    else :
        print(i)
        print('dsadsadsa')
        break

Sig_sets.rename(columns={'EKE_qiu_10_30_120_250':'EKE_qiu','EKE_qiu_10_30_120_250_ceemd_imf5':'EKE_qiu_ceemd_imf5',
                     'EKE_qiu_10_30_120_250_pc1':'EKE_qiu_pc1'},inplace=True)

Sig_sets.columns



Sig_sets = (Sig_sets -Sig_sets.mean())/(Sig_sets.std())

# =============================================================================
# Simple running mean (Center)
# ============================================================================


import pandas as pd

WY = 2
RM = Sig_sets.rolling(window=int(12*WY),center=True).mean()

# Removing RM ceemd
for i in RM.columns:
    if 'ceemd' in i:
        RM.drop(i,axis=1,inplace=True)
    else :
        RM.rename(columns={i:i+'_'+str(WY)+'Y_Rm'},inplace=True)

# Appending name"_Rm"

Sig_set = pd.concat([Sig_sets,RM],axis=1)

# Corr Matrix
Corr_Matrix = Sig_set[12:-12].corr()

print('!!!!!!!!!!!!!!!!!!!\nCorrcoef --> 1994~\n!!!!!!!!!!!!!!!!!!!')


Sig_set['date'] =  pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')



# =============================================================================
# Comparing Kuroshio
# =============================================================================

# w_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/fig2/'

def Pandas_plot(date,*args,**kwargs):
    import matplotlib.pyplot as plt
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['axes.linewidth'] = 3
    # plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.labelleft'] = True
    
    plt.figure(1,figsize=(16,10),dpi=80)
    ax = plt.gca()
    # print(args)
    # print('!!!!!!!!!!!!!!')
    a = kwargs.items()
    for i in a:
        print(i[1])
    for i, j in zip(args,kwargs.items()[1]):
        
        # print(i)
        # print('sadsadsadsadas')
        plt.plot(date,i,label='SSH Diff (2Y Runing mean)',color=j[1],linewidth=2.5)

    xtick_location = Sig_set.index.tolist()[::12*2]
    xtick_labels = Sig_set.date.tolist()[::12*2]
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, alpha=.7)
    # plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.yticks(fontsize=12, alpha=.7)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=.3)


a = {'1':1,'2':2,'3':3,'4':4}

# Sig_set.reset_index(inplace=True)
Pandas_plot(Sig_set['date'],Sig_set['PDO'],Sig_set['first_5'],'r')
plt.show()


plt.plot(Sig_set_set['date'],Sig_set_set['kuroshio_ceemd'])
plt.plot(Sig_set_set['date'],Sig_set_set['EKE_qiu_ceemdan'])


# =============================================================================
# EKE qiu
# =============================================================================
date =  pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')

plt.figure(1,figsize=(16,10),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')
# plt.plot(date,Sig_set.EKE_qiu.values,color=[.7,.7,.7],linewidth=2,label='SSH Diff')
# plt.plot(date,Sig_set.cmd_eke.values,label='running mean',color='darkred',linewidth=2.5)
plt.plot(date,Sig_set.EKE_qiu_eof.values,label='EOF pc1',color=[.85,.85,.85],linewidth=2.5)
# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.plot(Sig_set.date,Sig_set.EKE_qiu_Rm, label='SSH Diff (2Y Runing mean)',color='k',linewidth=2.5)
plt.plot(Sig_set.date,Sig_set.EKE_qiu_eof_Rm, label='EOF pc1 (2Y Runing mean)',color='b',linewidth=2.5)

plt.plot(Sig_set.date,Sig_set.EKE_qiu_ceemdan/5+.1, label='SSH Diff (ceemdan)',color='k',linewidth=3.5)
plt.plot(Sig_set.date,Sig_set.EKE_qiu_eof_ceemdan/5+.1, label='EOF pc1 (ceemdan)',color='b',linewidth=3.5)

# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.index.tolist()[::12*2]
xtick_labels = Sig_set.date.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=.3)
# if savefig:
    # plt.savefig(w_path+'EKE_qiu_normalized',dpi=150)
plt.show()



# =============================================================================
# 
# =============================================================================

plt.figure(1,figsize=(16,10),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.plot(Sig_set.date,Sig_set.first_5, label='Kuroshio_Rm (2Y Runing mean)',color=[0.8,0.8,0.8],linewidth=2.5)
plt.plot(Sig_set.date,Sig_set.second_3, label='Kuroshio_Rm (2Y Runing mean)',color=[0.6,0.6,0.6],linewidth=2.5)

plt.plot(Sig_set.date,Sig_set.first_5_2Y_Rm, label='Kuroshio_Rm (2Y Runing mean)',color='k',linewidth=2.5)
plt.plot(Sig_set.date,Sig_set.second_5_2Y_Rm, label='EKE_wang_pc1_2Y_Rm (2Y Runing mean)',color='darkred',linewidth=2.5)
plt.plot(Sig_set.date,Sig_set.second_3_2Y_Rm, label='EKE_wang_pc1_2Y_Rm (2Y Runing mean)',color='darkblue',linewidth=2.5)
# plt.plot(Sig_set.date,Sig_set.EKE_qiu_2Y_Rm, label='EKE qiu (2Y Runing mean)',color='darkred',linewidth=2.5)
# plt.plot(Sig_set.date,-Sig_set.WP_2Y_Rm, label='PDO (2Y Runing mean)',color='grey',linewidth=2.5)
# plt.plot(Sig_set.date,Sig_set.MEIv2_2Y_Rm, label='PDO (2Y Runing mean)',color='darkblue',linewidth=2.5)

# plt.plot(Sig_set.date,Sig_set.PDO_Rm, label='PDO (2Y Runing mean)',color='darkblue',linewidth=2.5)

# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.index.tolist()[::12*2]
xtick_labels = Sig_set.date.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=45, fontsize=22, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=22, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=.3)
# if savefig:
#     plt.savefig(w_path+'EKE_qiu_normalized',dpi=150)
plt.show()


# =============================================================================
# 
# =============================================================================

plt.figure(1,figsize=(16,10),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot

plt.plot(Sig_set.date,Sig_set.PDO_2Y_Rm, label='PDO (2Y Runing mean)',color='darkblue',linewidth=2.5,zorder=1)
plt.plot(Sig_set.date,Sig_set.EKE_qiu_10_30_120_250_2Y_Rm, label='eddy (2Y Runing mean)',color='darkred',linewidth=2.5,zorder=2)
plt.plot(Sig_set.date,Sig_set.WP_2Y_Rm, label='WP (2Y Runing mean)',color='g',linewidth=2.5,zorder=3)
plt.plot(Sig_set.date,-Sig_set.WP_2Y_Rm, label='-WP (2Y Runing mean)',color=[.8,.8,.8],linewidth=2.5,zorder=4)

plt.plot(Sig_set.date,Sig_set.ADT_index_2Y_Rm, label='SSH Diff (2Y Runing mean)',color='k',linewidth=2.5,zorder=0)
# plt.plot(Sig_set.date,Sig_set.first_5_2Y_Rm, label='EKE qiu (2Y Runing mean)',color='darkred',linewidth=2.5)

# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.index.tolist()[::12*2]
xtick_labels = Sig_set.date.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=30, fontsize=16, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=16, alpha=.7,fontweight='bold')
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='x', alpha=.3)
# if savefig:
    # plt.savefig(w_path+'EKE_qiu_normalized',dpi=150)
plt.show()



keys= Sig_set.columns



# =============================================================================
# 
#  Spatial Corr
# =============================================================================
from scipy import io

lonlat_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/data/qiu_2013/'
EKE_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/data/'
data_ = np.load(EKE_path+'EKE_10_30_120_250.npy')
dataset1 = data_[:]

tmp_lonlat = io.loadmat(lonlat_path+'eof.mat')
lon = tmp_lonlat['lon'][:].reshape(-1)
lat = tmp_lonlat['lat'][:].reshape(-1)

sig1 = Sig_set.first_5_2Y_Rm.values
Slicing_date = ['1993-01','2018-12']

tt,ii,jj = dataset1.shape
Corr_map = np.zeros([ii,jj])
for i in range(ii):
    for j in range(jj):
        tmp_dataset_sig = dataset1[:,i,j]
        tmp_Corr_set = pd.DataFrame({'tmp_dataset_sig':tmp_dataset_sig},index = Sig_set.date)
        tmp_Corr_set = tmp_Corr_set.rolling(window=12*WY,center=True).mean()
        tmp_Corr_set['sig1'] = sig1
        tmp = tmp_Corr_set.loc[Slicing_date[0]:Slicing_date[1]]
        Corr_map[i,j] = np.corrcoef(tmp.sig1[12:-12],tmp.tmp_dataset_sig[12:-12])[1,0]
    print(i)
    





# plt.contour(Corr_map,5)
plt.contourf(Corr_map,cmap='seismic')
plt.colorbar()
        


from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable



plt.figure(figsize=(26, 5))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=lat[0],urcrnrlat=lat[-1],\
            llcrnrlon=lon[0],urcrnrlon=lon[-1],resolution='i')
lon2, lat2 = np.meshgrid(lon,lat)
x, y = m(lon2, lat2)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(5.,33.,5.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=18,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=18,fontweight='bold',color='grey')
plt.title(' Corr 1993-01~2018-12', fontproperties='', position=(0.5, 1.0+0.07), fontsize=34,fontweight='bold')
# plt.suptitle(' UV & speed ',fontstyle='italic',position=(0.5, .92),fontsize=20)
CS = m.contour(x,y,Corr_map,6,colors='grey')
ax.clabel(CS,inline=True, fontsize=12,colors='k',fmt='%0.1f')
# CS.collections[6].set_linestyle('dashed')
CS.collections[6].set_linewidths(3)
CS.collections[6].set_color('k')



cs = m.pcolormesh(x,y,Corr_map,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-1,1)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 

h = plt.colorbar(label='',cax=cax);
# plt.savefig('F:/psi36/DATA/temp_var3/meanflow',bbox_inches='tight')

plt.show()


























