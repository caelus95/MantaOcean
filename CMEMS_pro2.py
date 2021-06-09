
"""
Created on Mon May 31 18:56:36 2021

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

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/CMEMS/' 
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

# w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_ERA/'

nc_list = np.sort([file for file in os.listdir(r_path1) if file.endswith(".nc")])


Sample_Data1 = xr.open_dataset(r_path1+nc_list[0])

data = xr.open_mfdataset(r_path1+'*.nc', parallel=True)
data_M = data.mean(dim='time')


Time = ['1994-01','2018-12'] 
minlon,maxlon = 112,180
fixed_lon = 150
minlat,maxlat = 5,45
mindepth, maxdepth = 0, 1200
    

data_s = data.loc[dict(time=slice(Time[0],Time[1]),latitude=slice(minlat,maxlat),
                        longitude=fixed_lon ,depth=slice(mindepth,maxdepth))]

data_s_M = data_s.mean(dim='time')

# =============================================================================
# figure setting
# =============================================================================
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "light"
plt.rcParams['axes.linewidth'] = 1
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True



# =============================================================================
# figure [0]
# =============================================================================


lat_s_M = data_s_M.latitude.values
lon_s_M = data_s_M.longitude.values
depth_s_M = -data_s_M.depth.values

lat_s_mesh,depth_s_mesh = np.meshgrid(lat_s_M,depth_s_M)

figdata01 = data_s_M.uo.values

# dens line

thetao_M02 =data_s_M.thetao.values
so_M02 =data_s_M.so.values

from gsw.density import rho 

dep,at = so_M02.shape
den_M02=np.zeros_like(so_M02)
for i in tqdm(range(at)):
    den_M02[:,i] = rho(so_M02[:,i],thetao_M02[:,i],-depth_s_M)

figdata02 = den_M02

# --------figure [vertical]---------
w_path_01 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/vertical_test/01/'

plt.figure(figsize=(15,11))
ax1=plt.gca()
plt.ylim(-1000,0)
plt.yticks(fontsize=22, alpha=1,)
plt.ylabel('$\it{depth(m)}$',fontsize=24,fontweight='light')
plt.xticks(fontsize=22, alpha=1)
plt.xlabel('$\it{latitude (\degree N)}$',fontsize=24,fontweight='light')
plt.title('Climatological U 1994~2018 ('+str(fixed_lon)+'$\degree$E) ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')

plt.axvline(x=17.5,color='k',linewidth=3,linestyle='--',alpha=.3)
plt.axvline(x=26.5,color='k',linewidth=3,linestyle='--',alpha=.3)

cs2 = plt.pcolormesh(lat_s_mesh,depth_s_mesh,figdata01,cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# plt.clim(-max_mapdata_2,max_mapdata_2)
plt.clim(-.3,.3)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label='m/s',cax=cax);
plt.savefig(w_path_01+'Climatological_150E' ,bbox_inches='tight')
plt.show()


# --------figure [vertical with dense]---------
w_path_02 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/vertical_test/02/withdens/'

plt.figure(figsize=(15,11))
ax1=plt.gca()
plt.ylim(-2000,0)
plt.yticks(fontsize=22, alpha=1,)
plt.ylabel('$\it{depth(m)}$',fontsize=24,fontweight='light')
plt.xticks(fontsize=22, alpha=1)
plt.xlabel('$\it{latitude (\degree N)}$',fontsize=24,fontweight='light')
plt.title('Climatological U 1994~2018 ('+str(fixed_lon)+'$\degree$E) ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
cs1 = plt.contour(lat_s_mesh,depth_s_mesh,figdata02,levels=10,colors='k',linestyles='-.',alpha=.8)
plt.clabel(cs1,fontsize=10,fmt='%1.1f')

# plt.grid(axis='x', alpha=.5)
plt.axvline(x=17.5,color='k',linewidth=3,linestyle='--',alpha=.3)
plt.axvline(x=26.5,color='k',linewidth=3,linestyle='--',alpha=.3)

cs2 = plt.pcolormesh(lat_s_mesh,depth_s_mesh,figdata01,cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# plt.clim(-max_mapdata_2,max_mapdata_2)
plt.clim(-.3,.3)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label='m/s',cax=cax);
plt.savefig(w_path_02+'Climatological_150E' ,bbox_inches='tight')
plt.show()



# =============================================================================
# figure [1]
# =============================================================================


figdata11 = data_s.uo.values

lat_10 = data_s.latitude.values
depth_10 = -data_s.depth.values

lat_10_mesh,depth_10_mesh = np.meshgrid(lat_10,depth_10)

time_10 = data_s.time.values




thetao_s12 =data_s.thetao.values
so_s12 =data_s.so.values

from gsw.density import rho 

t,dep,at = so_s12.shape
den_s12=np.zeros_like(so_s12)
for i in tqdm(range(t)):
    for j in range(at):
        den_s12[i,:,j] = rho(so_s12[i,:,j],thetao_s12[i,:,j],-depth_10)

figdata12 = den_s12



np.random.seed(10)
# -------figure-------
w_path_11 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/vertical_test/02/'


for i in range(10):
    tmp = np.random.randint(len(time_10))
    
    plt.figure(figsize=(15,11))
    ax1=plt.gca()
    plt.ylim(-2000,0)
    plt.yticks(fontsize=22, alpha=1,)
    plt.ylabel('$\it{depth(m)}$',fontsize=24,fontweight='light')
    plt.xticks(fontsize=22, alpha=1)
    plt.xlabel('$\it{latitude (\degree N)}$',fontsize=24,fontweight='light')
    plt.title(str(time_10[tmp])[:7]+' U ('+str(fixed_lon)+'$\degree$E) ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')

    plt.axvline(x=17.5,color='k',linewidth=3,linestyle='--',alpha=.3)
    plt.axvline(x=26.5,color='k',linewidth=3,linestyle='--',alpha=.3)
    
    cs2 = plt.pcolormesh(lat_10_mesh,depth_10_mesh,figdata11[tmp,:,:],cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
    # plt.clim(-max_mapdata_2,max_mapdata_2)
    plt.clim(-.3,.3)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    h = plt.colorbar(label='m/s',cax=cax);
    plt.savefig(w_path_11+str(time_10[tmp])[:7]+'_'+str(fixed_lon)+'E' ,bbox_inches='tight')
    plt.show()


# --------figure [vertical with dense]---------
    w_path_12 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_CMEMS/vertical_test/02/withdens/'

    
    plt.figure(figsize=(15,11))
    ax1=plt.gca()
    plt.ylim(-2000,0)
    plt.yticks(fontsize=22, alpha=1,)
    plt.ylabel('$\it{depth(m)}$',fontsize=24,fontweight='light')
    plt.xticks(fontsize=22, alpha=1)
    plt.xlabel('$\it{latitude (\degree N)}$',fontsize=24,fontweight='light')
    plt.title(str(time_10[tmp])[:7]+' U ('+str(fixed_lon)+'$\degree$E) ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=32,fontweight='bold')
    cs1 = plt.contour(lat_10_mesh,depth_10_mesh,figdata12[tmp,:,:],levels=10,colors='k',linestyles='-.',alpha=.8)
    plt.clabel(cs1,fontsize=10,fmt='%1.1f')
    
    # plt.grid(axis='x', alpha=.5)
    plt.axvline(x=17.5,color='k',linewidth=3,linestyle='--',alpha=.3)
    plt.axvline(x=26.5,color='k',linewidth=3,linestyle='--',alpha=.3)
    
    cs2 = plt.pcolormesh(lat_10_mesh,depth_10_mesh,figdata11[tmp,:,:],cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
    # plt.clim(-max_mapdata_2,max_mapdata_2)
    plt.clim(-.3,.3)
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    h = plt.colorbar(label='m/s',cax=cax);
    plt.savefig(w_path_12+str(time_10[tmp])[:7]+'_'+str(fixed_lon)+'E' ,bbox_inches='tight')
    plt.show()


































