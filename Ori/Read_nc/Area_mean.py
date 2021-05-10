#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:20:13 2020

@author: shamu
"""

import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/adt_Npacific_M.nc'
# w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/first_1_t.npy'
# w_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/lastest/figs/first_3.png'

Data = xr.open_dataset(r_path1)
adt = Data['adt']
# adt = adt - adt.mean(axis=0)
LAT,LON = Data['lat'], Data['lon']

nn = .15
mlat1,Mlat1,mlon1,Mlon1 = 24- nn, 24+ nn, 123.5- nn, 123.5+ nn
mlat2,Mlat2,mlon2,Mlon2 = 27.25- nn, 27.25+ nn, 123.25- nn, 123.25+ nn

# mlat1,Mlat1,mlon1,Mlon1 = 24- .5, 24+ .5, 123.5- .5, 123.5+ .5
# mlat2,Mlat2,mlon2,Mlon2 = 26.5- .5, 26.5+ .5, 123- .5, 123+ .5

# mlat1,Mlat1,mlon1,Mlon1 = 24- .5, 24+ .5, 123.5- .5, 123.5+ .5
# mlat2,Mlat2,mlon2,Mlon2 = 26.5- .5, 26.5+ .5, 123.- .5, 123.+ .5


point1 = adt.where( (adt.lat >= mlat1) & (adt.lat<=Mlat1) & (adt.lon >= mlon1) & (adt.lon <=Mlon1) ,
                   drop=True).mean(axis=(1,2),skipna=True)
point2 = adt.where( (adt.lat >= mlat2) & (adt.lat<=Mlat2) & (adt.lon >= mlon2) & (adt.lon <=Mlon2) ,
                   drop=True).mean(axis=(1,2),skipna=True)
Sig = point1 - point2
np.save(w_path1,Sig)

# plot boundary
Minlat1,Maxlat1,Minlon1,Maxlon1 = 23,28,120,125

#plot
plt.figure(figsize=(9, 11))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=Minlat1,urcrnrlat=Maxlat1,
            llcrnrlon=Minlon1,urcrnrlon=Maxlon1,resolution='i')
m.drawmapboundary(linewidth=3)
lon2, lat2 = np.meshgrid(LON,LAT)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(19,31,1),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(120.,130.,1.5),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
# plt.title(' Referrenced points ', position=(.5, 5.0+0.1),fontweight='bold',fontsize=46)
m.plot([mlon1,mlon1,Mlon1,Mlon1,mlon1],[mlat1,Mlat1,Mlat1,mlat1,mlat1],color=[.6,0,0])
m.plot([mlon2,mlon2,Mlon2,Mlon2,mlon2],[mlat2,Mlat2,Mlat2,mlat2,mlat2],color=[.6,0,0])
m.scatter([122.2,122.4,122.6,122.8,123],[24.5,24.4,24.3,24.2,24.1],
          color='k',marker='<',s=200)

m.plot([123.5,123.25],[24,27.25],color=[.1,.1,.6],linewidth=4,linestyle='-.',marker='*',
       markerfacecolor=[.1,.1,.6],markersize=25,markeredgecolor=[.1,.1,.6])
m.plot([123.5,123.],[24,26.5],color=[.6,.1,.1],linewidth=4,linestyle='-.',marker='*',
       markerfacecolor=[.6,.1,.1],markersize=25,markeredgecolor=[.6,.1,.1])
# plt.savefig(w_path2, dpi=200)
plt.show()
     




