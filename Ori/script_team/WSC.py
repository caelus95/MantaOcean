# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:35:59 2020

@author: psi36
"""


import numpy as np
from scipy import io
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


'''Data'''

llcrnrlat = -20 
urcrnrlat = 60  
llcrnrlon = 112 
urcrnrlon = 300 

mat_dict = io.loadmat('F:/psi36/DATA/Kuroshio_NEC/WSC_pacific')

lat_rgnl = mat_dict['lat_rgnl'][:]
lon_rgnl = mat_dict['lon_rgnl'][:]

lat_co = mat_dict['lat_co'][:]
lon_co = mat_dict['lon_co'][:]

lat = mat_dict['lat'][:]
lon = mat_dict['lon'][:]

Ac,Bc = np.meshgrid(lon_rgnl,lat_rgnl)

hyr_mean_u = mat_dict['hyr_mean_u'][:]
hyr_mean_v = mat_dict['hyr_mean_v'][:]

lyr_mean_u = mat_dict['lyr_mean_u'][:]
lyr_mean_v = mat_dict['lyr_mean_v'][:]

hyr_curlZ_mean_at = mat_dict['hyr_curlZ_mean_at'][:]
lyr_curlZ_mean_at = mat_dict['lyr_curlZ_mean_at'][:]

hyr_curlZ = mat_dict['hyr_curlZ'][:]*10**8
lyr_curlZ = mat_dict['lyr_curlZ'][:]*10**8
    

'''plot'''

# plt.plot(hyr_curlZ_mean_at,lat_rgnl,color='green',libewidth=5)

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(hyr_curlZ_mean_at, np.sort(lat_rgnl,axis=0), label='Positive')  # Plot some data on the axes.
ax.plot(lyr_curlZ_mean_at, np.sort(lat_rgnl,axis=0), label='Negative')  # Plot more data on the axes...
ax.set_xlabel('10^-8 N / m^-3')  # Add an x-label to the axes.
ax.set_ylabel('latitude')  # Add a y-label to the axes.
ax.set_title("Wind Stress Curl Anomaly")  # Add a title to the axes.
ax.legend()  # Add a legend.



'''basemap'''


plt.figure(figsize=(25, 20))
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
plt.title(' Negative years ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
cs = m.pcolormesh(Ac,Bc,lyr_curlZ,cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
plt.clim(-.1,.1)
#m.quiver(Acp,Bcp,u_s,v_s,scale=7,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 

h = plt.colorbar(label='10^-8 N / m^3',cax=cax);
plt.savefig('F:/psi36/kuroshi/temp/Nega1',bbox_inches='tight')

plt.show()




















