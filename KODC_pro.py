#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:06:40 2021

@author: caelus
"""
from matplotlib import dates
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/KODC/rawdata/'

Data = pd.read_csv(r_path+'all2.csv') 

Data.drop(['Region'],axis=1,inplace=True)

# Remove QC_level==4
Data.drop( Data[ Data['Temp_flag'] == 4 ].index, inplace=True)

Data_315 = Data[Data['Vessle'] == 315]
Data_316 = Data[Data['Vessle'] == 316]
# Data_317 = Data[Data['Vessle'] == 317] # 317 data starts at 2000

Data_315.index = pd.to_datetime(Data_315['Time'], format='%m/%d/%Y')
Data_316.index = pd.to_datetime(Data_316['Time'], format='%m/%d/%Y')
# Data_317.index = pd.to_datetime(Data_317['Time'], format='%m/%d/%Y')

# Remove head & tail (bad Sampling)

Data_315.loc['1994',:],Data_315.loc['2020',:] = np.nan,np.nan
Data_316.loc['1994',:],Data_316.loc['2020',:] = np.nan,np.nan


Data_315.drop(Data_315[(Data_315['Lon'] == 125.287) | (Data_315['Lon'] == 125.587)\
                       ].index,inplace=True)


# Data yearly mean
Data_315_Y = Data_315.resample('y').mean()
Data_316_Y = Data_316.resample('y').mean()




# Plot temp yearly mean
Data_315_Y['Temp'].plot()
Data_316_Y['Temp'].plot()


# Count smaple number
tmp_Data_315 = Data_315[Data_315['Depth'] == 0]
tmp_Data_316 = Data_316[Data_316['Depth'] == 0]

Data_315_Month_count = pd.to_datetime(tmp_Data_315['Temp'].index, format='%M')\
    .strftime('%m').value_counts().sort_index()
    
Data_316_Month_count = pd.to_datetime(tmp_Data_316['Temp'].index, format='%M')\
    .strftime('%m').value_counts().sort_index()

# ----figure Sample number ----
Sample_num_315 = str(Data_315_Month_count.sum())
Sample_num_316 = str(Data_316_Month_count.sum())


fig, (ax1, ax2) = plt.subplots(2,figsize=(8,6), sharex=True)
plt.title('dsa')
ax1.set_title('St. 315 (1995~2019)',fontweight='bold',fontsize=20)
ax1.bar(Data_315_Month_count.index,Data_315_Month_count,color='k')
ax1.set_ylabel('Sample number [N]',fontsize=16)
# ax1.set_xlabel('Month')
ax1.grid(True,axis='y',linestyle='--')
ax1.tick_params(axis='both',labelsize=16)

ax1.text('01',270,'N : '+Sample_num_315,verticalalignment='top',\
         horizontalalignment='center',fontsize=12)

ax2.set_title('St. 316 (1995~2019)',fontweight='bold',fontsize=20)
ax2.bar(Data_316_Month_count.index,Data_316_Month_count,color='k')
ax2.set_ylabel('Sample number [N]',fontsize=16)
ax2.set_xlabel('Month',size=16)

ax2.grid(True,axis='y',linestyle='--')
ax2.tick_params(axis='both',labelsize=16)
ax2.text('01',270,'N : '+Sample_num_316,verticalalignment='top',\
         horizontalalignment='center',fontsize=12)

plt.tight_layout()
plt.show()


# ----figure Yearly mean time series ----

T1 = Data_315_Y.index.strftime('%Y')
Data_315_Y_mean = Data_315_Y['Temp'].mean()
Data_316_Y_mean = Data_316_Y['Temp'].mean()
Data_315_Y_std = Data_315_Y['Temp'].std()
Data_316_Y_std = Data_316_Y['Temp'].std()

figdata1 = Data_315_Y['Temp'] - Data_315_Y_mean
figdata2 = Data_316_Y['Temp'] - Data_316_Y_mean



plt.figure(1,figsize=(16,6),dpi=80)
ax = plt.gca()
  
plt.title('a) Temperature anomaly (Yearly mean)', fontproperties='',loc='left',pad=15,  fontsize=20,fontweight='bold')
plt.plot(T1,figdata1,color='k',linewidth=3,zorder=10)
plt.plot(T1,figdata2,color='darkred',linewidth=3,zorder=9)

plt.scatter(T1,figdata1,
            color='k',marker='o',s=200,zorder=18, label='St.315 (32.5$\degree$N)')
plt.scatter(T1,figdata2,
            color='darkred',marker='o',s=200,zorder=17, label='St.316 (32$\degree$N)')

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)

# Decoration
# plt.ylim(50,750)
xtick_location = T1.tolist()[::2]
xtick_labels = T1.tolist()[::2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower right',fontsize=16)
plt.grid(axis='x', alpha=.6)
plt.tight_layout()
# plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
plt.show()



# =============================================================================
# 
# =============================================================================
Data_315['T'] = pd.to_datetime(Data_315.index, format='%Y-%M-%D')#.strftime('%Y-%m')
Data_316['T'] = pd.to_datetime(Data_316.index, format='%Y-%M-%D')#.strftime('%Y-%m')

Data_315_v = Data_315[['Temp','Depth','Lon','T']].dropna()
Data_316_v = Data_316[['Temp','Depth','Lon','T']].dropna()

Data_315_4xr = Data_315_v.groupby(['Depth','T','Lon']).mean()
Data_316_4xr = Data_316_v.groupby(['Depth','T','Lon']).mean()

Data_315_xr = Data_315_4xr.to_xarray()
Data_316_xr = Data_316_4xr.to_xarray()


Data_315_xr = Data_315_xr.resample(T="1YS").mean(dim="T")
Data_316_xr = Data_316_xr.resample(T="1YS").mean(dim="T")

Data_315_xr = Data_315_xr.loc[dict(Depth=slice(0,300),Lon=slice(124,130))]
Data_316_xr = Data_316_xr.loc[dict(Depth=slice(0,100),Lon=slice(124,130))]

Data_315_xr_a = Data_315_xr - Data_315_xr.mean(dim='T')
Data_316_xr_a = Data_316_xr - Data_316_xr.mean(dim='T')

Varli_315 = Data_315_xr_a.Temp.mean(dim=['Lon','Depth'])
Varli_316 = Data_316_xr_a.Temp.mean(dim=['Lon','Depth'])

# --------------figure Validation Time series ----------
T1 = T1[1:-1]
plt.figure(1,figsize=(16,6),dpi=80)
ax = plt.gca()
  
plt.title('a) Temperature anomaly (Yearly mean)', fontproperties='',loc='left',pad=15,  fontsize=20,fontweight='bold')
plt.plot(T1,Varli_315,color='k',linewidth=3,zorder=10)
plt.plot(T1,Varli_316,color='darkred',linewidth=3,zorder=9)

plt.scatter(T1,Varli_315,
            color='k',marker='o',s=200,zorder=18, label='St.315 (32.5$\degree$N)')
plt.scatter(T1,Varli_316,
            color='darkred',marker='o',s=200,zorder=17, label='St.316 (32$\degree$N)')

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)

# Decoration
# plt.ylim(50,750)
xtick_location = T1.tolist()[1::2]
xtick_labels = T1.tolist()[1::2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower right',fontsize=16)
plt.grid(axis='x', alpha=.6)
plt.tight_layout()
# plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
plt.show()






A = Data_315_xr_a.T.values
B = np.arange(25)
C = pd.DataFrame({'T':A,'B':B})




A1  = Data_315_xr_a.Temp[1] +Data_315_xr_a.Temp[2] +Data_315_xr_a.Temp[3] +Data_315_xr_a.Temp[8] +\
    Data_315_xr_a.Temp[9] +Data_315_xr_a.Temp[13] +Data_315_xr_a.Temp[14] +Data_315_xr_a.Temp[21]
A1 = A1/8
x_m,y_m = np.meshgrid(A1.Lon,A1.Depth)

plt.pcolormesh(x_m,-y_m,A1,cmap=plt.cm.get_cmap('seismic'))


B1  = Data_315_xr_a.Temp[5] +Data_315_xr_a.Temp[6] +Data_315_xr_a.Temp[10] +Data_315_xr_a.Temp[11] +\
    Data_315_xr_a.Temp[17] +Data_315_xr_a.Temp[18] +Data_315_xr_a.Temp[19]
B1 = B1/7
x_m,y_m = np.meshgrid(B1.Lon,B1.Depth)

plt.pcolormesh(x_m,-y_m,B1,cmap=plt.cm.get_cmap('seismic'))



for ii in range(25):
    plt.pcolormesh(x_m,-y_m,tmp_t.Temp[ii,:,:],cmap=plt.cm.get_cmap('seismic'))
    plt.title(tmp_t.T.values[ii])
    plt.colorbar()
    plt.clim(-3.,3.)
    plt.show()


tmp_d = Data_315_xr.Depth.values
tmp_l = Data_315_xr.Lon.values
tmp_t = np.arange(104)

V = Data_315_xr.Temp.values
V = V.transpose([1,0,2])

from scipy.interpolate import interpn,LinearNDInterpolator

T,D,L = np.meshgrid(tmp_t,tmp_d,tmp_l)
points = (tmp_t,tmp_d,tmp_l)
xi,yi,zi = tmp_t,np.arange(0,61,5),np.arange(124,127.7,.2)
TEST = interpn(points, V, (xi,yi,zi))



from scipy.interpolate import RegularGridInterpolator as rgi
my_interpolating_function = rgi((tmp_t,tmp_d,tmp_l), V)
Vi = my_interpolating_function(np.array([xi,yi,zi]).T)



Vi = interpn((tmp_t,tmp_d,tmp_l), V, np.array([xi,yi,zi]).T)




interp = LinearNDInterpolator(list(zip(tmp_t, tmp_d,tmp_l)), V)

A = tmp_t.Temp.mean(dim=['Lon','Depth'])

plt.plot(A)


