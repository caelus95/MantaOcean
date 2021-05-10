# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:46:45 2020

@author: psi36
"""

from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import os
from numpy import hstack


WSC = io.loadmat('F:/psi36/DATA/Kuroshio_NEC/WesternBound_ERA5_windstresscurl')['WindStressCurl'][:]
lonlat = io.loadmat('F:/psi36/DATA/Kuroshio_NEC/WesternBound_ERA5_lonlat')

WSC_mean = np.nanmean(WSC,axis=0)
WSC_anomaly = WSC - WSC_mean

lon,lat = lonlat['lon'][0],  lonlat['lat'][0]
lon_co, lat_co = lonlat['lon_co'][0], lonlat['lat_co'][0]

lon_rgnl, lat_rgnl = lon[lon_co], lat[lat_co]

var_composite = composite1(WSC_anomaly)



hyr_WSC = var_composite['hyr']
lyr_WSC = var_composite['lyr']

Ac,Bc = np.meshgrid(lon_rgnl,lat_rgnl)

'''
===============================================================================
'''

plt.figure(figsize=(20, 10))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,\
            llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='i')
lon2, lat2 = np.meshgrid(lon_rgnl,lat_rgnl)
x, y = m(lon2, lat2)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,1.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,2.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title('WSC Positive years ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
cs = m.pcolormesh(Ac,Bc,WSC_hyr,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
plt.clim(-.000000001,.000000001)
#m.quiver(Acp,Bcp,u_s,v_s,scale=7,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 

h = plt.colorbar(label='m/s',cax=cax);
#plt.savefig('F:/2',bbox_inches='tight')

plt.show()


