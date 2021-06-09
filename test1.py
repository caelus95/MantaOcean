
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


r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/EKE/qiu_wang/'

# r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/adt_8_30_112_180_M.nc'


# w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/fig2/'


Sig_set,Corr_map,Annual_mean = sig_pro(r_path1,['1993-01-01',324,300],Standard=True)

# data, coor = nc2npy(r_path2)
# time,lon,lat = coor[0],coor[1],coor[2]
# adt = data[0]

adt = np.load(r_path2+'EKE_10_30_120_250.npy') 
lon = np.load(r_path2+'lon_10_30_120_250.npy') 
lat = np.load(r_path2+'lat_10_30_120_250.npy') 


# adt = adt - np.mean(adt,axis=0)

Coef1, p_values1 = linearRegress4Cube(Sig_set.ADT_index_2Y_Rm.dropna()[:-1],adt[12:-12,:,:],['1994-01','2018-12'])
Coef2, p_values2 = linearRegress4Cube(Sig_set.ADT_index_2Y_Rm.dropna()[:-1],adt[12:-12,:,:],['1994-01','2005-10'])
Coef3, p_values3 = linearRegress4Cube(Sig_set.ADT_index_2Y_Rm.dropna()[:-1],adt[12:-12,:,:],['2005-11','2011-01'],method='1')
Coef4, p_values4 = linearRegress4Cube(Sig_set.ADT_index_2Y_Rm.dropna()[:-1],adt[12:-12,:,:],['2011-02','2018-12'])


Coef5, p_values3 = linearRegress4Cube(Sig_set.ADT_index_2Y_Rm.dropna()[:-1],adt[12:-12,:,:],['2005-11','2013-03'])
Coef6, p_values4 = linearRegress4Cube(Sig_set.ADT_index_2Y_Rm.dropna()[:-1],adt[12:-12,:,:],['2013-04','2018-12'])

Minlat,Maxlat,Minlon,Maxlon = 8,30,112,180


Minlat,Maxlat,Minlon,Maxlon = 10,30,120,250

m,M = np.nanmin(Coef3),np.nanmax(Coef3)


plt.figure()




plt.figure(figsize=(18, 4))
plt.title('2005-11 ~ 2011-01',fontsize=38,fontweight='bold')
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=Minlat,urcrnrlat=Maxlat,
            llcrnrlon=Minlon,urcrnrlon=Maxlon,resolution='i')
m.drawmapboundary(linewidth=3)
# lon2, lat2 = np.meshgrid(LON,LAT)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines(),
m.drawparallels(np.arange(-10,60,10),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(120.,262.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
# plt.title(' Positive phase (sla)', position=(.5, 5.0+0.1),fontweight='bold',fontsize=46)
# plt.suptitle(' 
x, y = m(lon,lat)
m.pcolormesh(x,y,Coef33,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-0.2,.2)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label=f'',cax=cax);
# plt.savefig(w_path1+'EKE_regress_qiu_201102_201812', dpi=200)
plt.show()



plt.figure(figsize=(18, 4))
plt.title('1994-01 ~ 2005-10',fontsize=38,fontweight='bold')
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=Minlat,urcrnrlat=Maxlat,
            llcrnrlon=Minlon,urcrnrlon=Maxlon,resolution='i')
m.drawmapboundary(linewidth=3)
# lon2, lat2 = np.meshgrid(LON,LAT)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines(),
m.drawparallels(np.arange(-10,60,10),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(120.,262.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
# plt.title(' Positive phase (sla)', position=(.5, 5.0+0.1),fontweight='bold',fontsize=46)
# plt.suptitle(' 
x, y = m(lon,lat)
m.pcolormesh(x,y,Coef2,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-.05,.05)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label=f'',cax=cax);
# plt.savefig(w_path1+'EKE_regress_qiu_201102_201812', dpi=200)
plt.show()


plt.figure(figsize=(18, 4))
plt.title('2005-11 ~ 2013-03',fontsize=38,fontweight='bold')
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=Minlat,urcrnrlat=Maxlat,
            llcrnrlon=Minlon,urcrnrlon=Maxlon,resolution='i')
m.drawmapboundary(linewidth=3)
# lon2, lat2 = np.meshgrid(LON,LAT)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines(),
m.drawparallels(np.arange(-10,60,10),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(120.,262.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
# plt.title(' Positive phase (sla)', position=(.5, 5.0+0.1),fontweight='bold',fontsize=46)
# plt.suptitle(' 
x, y = m(lon,lat)
m.pcolormesh(x,y,Coef5,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-.05,.05)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label=f'',cax=cax);
# plt.savefig(w_path1+'EKE_regress_qiu_201102_201812', dpi=200)
plt.show()



