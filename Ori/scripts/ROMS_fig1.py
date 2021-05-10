import numpy as np
import pandas as pd
import xarray as xr
import os 
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt


Output_path = '/home/caelus/dock_1/Working_hub/test_dep/room_1/His/' # his / avg ...

list_dir = np.sort([file for file in os.listdir(Output_path) if file.endswith('.nc')])

i=list_dir[0]
#for i in list_dir:
test_file = xr.open_dataset(Output_path+i,decode_times=False)
time = test_file['ocean_time']
temp = test_file['temp']
slice_temp = temp[:,-1,:,:]

lon_rho,lat_rho = slice_temp.lon_rho.values, slice_temp.lat_rho.values

len(slice_temp.ocean_time.values)
print(slice_temp.ocean_time.values)
t,at,on = slice_temp.values.shape
# =============================================================================
# 
# =============================================================================

# for i in range(t):
i =0 

plt.rcParams["font.size"] = 50
plt.rcParams['font.weight'] = 'bold'
plt.style.use('seaborn-darkgrid')
# plt.rcParams['font.family'] = 'New Century Schoolbook'
plt.rc('font',size=32)
plt.rcParams['axes.labelweight']='bold'

plt.figure(figsize=(20, 10))
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=lat_rho[0,0],urcrnrlat=lat_rho[-1,-1],
            llcrnrlon=lon_rho[0,0],urcrnrlon=lon_rho[-1,-1],resolution='c',width=2)
m.drawmapboundary(linewidth=3)
# lon2, lat2 = np.meshgrid(lon_rgnl,lat_rgnl)
x, y = m(lon_rho, lat_rho)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(lat_rho[0,0],lat_rho[-1,-1],10),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(round(lon_rho[0,0]+5,-1),lon_rho[-1,-1],20.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
                
plt.title('  ', position=(0.5, 1.0+0.05),fontweight='bold',fontsize=46)
# plt.suptitle(' UV (anomaly) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
   # seismic
cs = m.pcolormesh(x,y,slice_temp[i,:,:].values,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
plt.clim(0,32)
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
'''
plt.show()

if Save_fig_info:
    import inspect
    inspect.getsource(object)
'''