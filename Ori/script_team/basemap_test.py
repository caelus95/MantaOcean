# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:04:11 2020

@author: psi36
"""






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
plt.title(' Mean flow ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
plt.suptitle(' UV & speed ',fontstyle='italic',position=(0.5, .92),fontsize=20)
cs = m.pcolormesh(Ac,Bc,speed1,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
plt.clim(0,.5)
m.quiver(Acp,Bcp,u_s,v_s,scale=7,headwidth=5,headaxislength=10,headlength=10)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 

h = plt.colorbar(label='m/s',cax=cax);
plt.savefig('F:/psi36/DATA/temp_var3/meanflow',bbox_inches='tight')

plt.show()


