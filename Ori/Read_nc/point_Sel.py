#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 14:39:18 2020

@author: shamu
"""

import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np



r_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/adt_Npacific_M.nc'
# w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/yan_sun_points/'
# w_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/figs/Yan_sun.png'

Data = xr.open_dataset(r_path1)

LAT = Data['lat']
LON = Data['lon']

# =============================================================================
# Find coordinates
# =============================================================================

def find_coor(mlat,Mlat,mlon,Mlon,LAT,LON):
    import numpy as np
    lat_co = np.where( (LAT >= mlat) & (LAT <= Mlat))[0]
    lon_co = np.where( (LON >= mlon) & (LON <= Mlon))[0]
    return lat_co,lon_co

lat_co1,lon_co1 = find_coor(23.9,24.3,123.4,123.7,LAT,LON)
lat_co2,lon_co2 = find_coor(27,27.3,123.1,123.3,LAT,LON)

# =============================================================================
# Cartesian multiple (paring)
# =============================================================================

p1,q1 = np.zeros(2),np.zeros(2)
p1[0],p1[1],q1[0],q1[1] = lat_co1,lat_co1-1,lon_co1,lon_co1-1

p2,q2 = np.zeros(2),np.zeros(2) 
p2[0],p2[1],q2[0],q2[1] = lat_co2,lat_co2-1,lon_co2,lon_co2-1

point_array1 = []
point_array2 = []

for i in range(2):
    for j in range(2):
        point_array1.append([p1[i],p2[j]]) 
        point_array2.append([q1[i],q2[j]])

point_matrix = []
for i in range(4):
    for j in range(4):
        point_matrix.append([point_array1[i],point_array2[j]])
point_matrix = np.array(point_matrix).astype('int')


# =============================================================================
# ploting & save 
# =============================================================================

Minlat,Maxlat,Minlon,Maxlon = 22,28,120,124.5

plt.figure(figsize=(9, 11))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=Minlat,urcrnrlat=Maxlat,
            llcrnrlon=Minlon,urcrnrlon=Maxlon,resolution='i')
m.drawmapboundary(linewidth=3)
lon2, lat2 = np.meshgrid(LON,LAT)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(19,31,2),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(120.,130.,2.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
plt.title(' Referrenced points ', position=(.5, 5.0+0.1),fontweight='bold',fontsize=46)
# plt.suptitle(' 
n=1
for i in point_matrix:
    x1, y1 = m(LON[i[1]],LAT[i[0]])
    m.plot(x1, y1,color=[.8,.8,.8],marker='o',markeredgecolor='k',
           markerfacecolor='k',markersize=8)
    point1 = Data['adt'][:,i[0][0],i[1][0]]
    point2 = Data['adt'][:,i[0][1],i[1][1]]
    dif = point1 - point2
    dif = np.squeeze(dif.data)
    # np.save(w_path1+f'{n:02d}',dif)
    print(LAT[i[0]].values,LON[i[1]].values)
    n+=1
m.plot([123.5,123],[24,27],color=[.6,.1,.1],linewidth=4,linestyle='-.',marker='*',markerfacecolor=[.6,.1,.1],
       markersize=25)
# m.scatter([123.5,123],[24,27],marker='*',s=250,color=[.6,.1,.1])
# m.plot([123.5,123],[24,27],color='k',linestyle='-.',linewidth=3.5)
m.scatter([122.2,122.4,122.6,122.8,123],[24.5,24.4,24.3,24.2,24.1],
          color='k',marker='<',s=200)
# plt.savefig(w_path2, dpi=200)
plt.show()
     
# np.save(w_path1+'point_matrix.npy',point_matrix)


'''
point1 = Data['adt'][:,i[0][0],i[1][0]]
point2 = Data['adt'][:,i[0][1],i[1][1]]
dif = point1 - point2
dif = np.squeeze(dif.data)
np.save(w_path1,dif)
'''








