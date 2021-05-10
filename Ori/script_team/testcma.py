# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:48:51 2019

@author: psi36
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 02:22:40 2019

@author: manta36
"""
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

'''
llcrnrlat=-60
urcrnrlat=60
llcrnrlon=0.1
urcrnrlon=359
t_len = 300
'''
## basemap
var = np.nanmean(var_composite['hyr'],axis=0)

def colorbarAXIS(var):
    MAX_data = np.max(var)
    MIN_data = np.min(var)
    M = np.max([abs(MAX_data),abs(MIN_data)])
    L1 = M/4 ; L2 = -M/4
    levels = np.arange(-M,M+L,L)
    return levels, L

levels,L = colorbarAXIS(var)

plt.figure(figsize=(20, 10))

m = Basemap(projection='cyl',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,\
            llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='c')
lon2, lat2 = np.meshgrid(lon_rgnl,lat_rgnl)
x, y = m(lon2, lat2)
m.fillcontinents(color='black',lake_color='navy')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold')
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold')
plt.title(' DJF_lyr', fontproperties='', position=(0.5, 1.0+0.05), fontsize=40,fontweight='bold')
cs=m.contourf(x,y,var,20,cmap=plt.cm.get_cmap('jet'))

ax = plt.gca()
'''
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)

cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#plt.colorbar(cax, ax=ax[-2.5,2.5])
#cax.set_clim([-2.5, 2.5])
#label 

h = plt.colorbar(cs,label='mm/day');
'''
#plt.colorbar(cs)
plt.colorbar(ticks=levels)
ax.set_yticklabels(levels)

plt.show()

