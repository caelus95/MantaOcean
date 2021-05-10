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
#speed_hyr = (np.nanmean(hyr_u**2,axis=0) + np.nanmean(hyr_v**2,axis=0))**.5
#speed_lyr = (np.nanmean(lyr_u**2,axis=0) + np.nanmean(lyr_v**2,axis=0))**.5




plt.figure(figsize=(20, 10))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
            llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,resolution='i')
lon2, lat2 = np.meshgrid(lon_rgnl,lat_rgnl)
x, y = m(lon2, lat2)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,1.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,2.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=13,fontweight='bold',color='grey')
plt.title(' Positive years ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
plt.suptitle(' UV (anomaly) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
cs = m.pcolormesh(Ac,Bc,np.nanmean(sla[LYR_period[:,17]][:],axis=0),cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
plt.clim(-.1,.1)
#m.quiver(Ac,Bc,np.nanmean(hyr_u,axis=0),np.nanmean(hyr_v,axis=0),scale=1,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 

h = plt.colorbar(label='m/s',cax=cax);
##plt.savefig('F:/psi36R/kuroshi/Kuroshio_ori_posi_a_af',bbox_inches='tight')

plt.show()


