# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:24:58 2020

@author: psi36
"""

#var=io.loadmat('F:/psi36/DATA/temp_var3/EastP_adt')



# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 02:22:40 2019

@author: manta36
"""
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

'''
llcrnrlat=0
urcrnrlat=60
llcrnrlon=0.1
urcrnrlon=359
t_len = 300
'''
## basemap

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
plt.title(' Mean ssh ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
plt.suptitle(' ADT-H  (Absolute Dynamic Topology Height) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
cs = m.contourf(x,y,np.nanmean(adt,axis=0),20,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
plt.clim(.7,1.6)
#m.quiver(x,y,U,V,scale=7)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 

h = plt.colorbar(label='m',cax=cax);

CS = ax.contour(x, y,np.nanmean(adt,axis=0),20,colors='grey')
ax.clabel(CS, inline=1, fontsize=14,colors='k')

#CS.collections[9].set_linestyle('dashed')
CS.collections[7].set_linewidths(3)
CS.collections[7].set_color('k')

CS.collections[8].set_linestyle('dashed')
CS.collections[8].set_linewidths(3)
CS.collections[8].set_color('red')

CS.collections[9].set_linestyle('dashed')
CS.collections[9].set_linewidths(3)
CS.collections[9].set_color('b')

#CS.collections[12].set_linestyle('dashed')
CS.collections[10].set_linewidths(3)
CS.collections[10].set_color('k')


plt.savefig('F:/psi36/DATA/temp_var3/adt_contour_mean',bbox_inches='tight')

plt.show()





