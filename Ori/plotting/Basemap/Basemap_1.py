# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 02:22:40 2019

@author: manta36
"""
import sys
sys.path.append('/home/shamu/mangrove1/Working_hub/LGnDC_dep/python_cent/pycode_team/')
from psi_package.psi_tools import psi_load_grid_data_zip
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
'''
llcrnrlat=0
urcrnrlat=60
llcrnrlon=0.1
urcrnrlon=359
t_len = 300
'''

# import matplotlib.font_manager as fm
# fm.get_fontconfig_fonts()
# font_location = 'C:/Windows/Fonts/New Century Schoolbook/New Century Schoolbook Regular.ttf'
# # font_location = 'C:/Windows/Fonts/NanumGothic.ttf' # For Windows
# font_name = fm.FontProperties(fname=font_location).get_name()
# matplotlib.rc('font', family=font_name)

Data_Dir1 = ''
Pcolor_Data = '' # empty if no data
Data_Dir2 = ''
Quiver_Data = '' # empty if no data
save_dir = ''

Save_fig_info = False


lat_rgnl,lon_rgnl,lat_co,lon_co,lat_F,lon_F,Data = psi_load_grid_data_zip(Data_Dir1+Pcolor_Data)

Data_re = Data
Maxlat = lat_rgnl[-1] 
Minlat = lat_rgnl[0]
Maxlon = lon_rgnl[-1]
Minlon = lon_rgnl[0]


Rc_Parameters = []
Parameters = []

plt.rcParams["font.size"] = 50
plt.rcParams['font.weight'] = 'bold'
plt.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'New Century Schoolbook'
plt.rc('font',size=32)
plt.rcParams['axes.labelweight']='bold'


plt.figure(figsize=(9.7, 9))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=Maxlat,urcrnrlat=Minlat,
            llcrnrlon=Minlon,urcrnrlon=Maxlon,resolution='i',width=2)
m.drawmapboundary(linewidth=3)
lon2, lat2 = np.meshgrid(lon_rgnl,lat_rgnl)
x, y = m(lon2, lat2)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(19,31,2),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(120.,130.,2.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
plt.title(' ET speed EOF ', position=(0.5, 1.0+0.05),fontweight='bold',fontsize=46)
# plt.suptitle(' UV (anomaly) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
if Pcolor_Data:
    
    cs = m.pcolormesh(x,y,Data_re,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.pcolor(X,Y,data, hatch=".",alpha=0)
    plt.clim(-.07,.07)
    ax.tick_params(labelcolor='k',labelsize=22,pad=15,grid_linewidth=2)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=18,width='bold')
    cax.set_ylabel('',{'fontsize':32,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='m/s',cax=cax);
    # plt.savefig('E:/psi36/MATLAB_linux/KUROSHIO/Figs/ET_speed_EOF',bbox_inches='tight',dpi=150)

# if Quiver_Data:
#     m.quiver(Ac,Bc,u_s,v_s,scale=5,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25])

plt.show()

if Save_fig_info:
    import inspect
    inspect.getsource(object)

















