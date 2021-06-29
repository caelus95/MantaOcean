
PKG_path = '/home/caelus/dock_1/Working_hub/LGnDC_dep/python_cent/MantaPKG/'

import sys 
sys.path.append(PKG_path)

from Manta_Signals.procc_index import sig_pro, linearRegress4Cube
from Manta_Signals.utility import nc2npy
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/CDS/'
# r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_EKE/data/'
# r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/'


data = xr.open_mfdataset(r_path1+'*.nc', parallel=True)

minlat, maxlat =23.3,23.7
minlon, maxlon =120, 180
data = data.loc[dict(latitude=slice(minlat,maxlat),longitude=slice(minlon,maxlon),nv=1)]

data = data.drop(['crs','lat_bnds','lon_bnds','err','sla','adt'])

data_a = data - data.mean(dim='time')


# data_a_6M = data_a_6M.mean(dim='latitude')
def MantaCurl2D(u,v,dx=28400.0,dy=28400.0 ):
    import numpy as np
    '''
    dx = 28400.0 # meters calculated from the 0.25 degree spatial gridding 
    dy = 28400.0 # meters calculated from the 0.25 degree spatial gridding 
    '''
    u_T = u.transpose([1,0])
    v_T = v.transpose([1,0])

    
    du_dx, du_dy = np.gradient(u_T, dx,dy)
    dv_dx, dv_dy = np.gradient(v_T, dx,dy)

    curl = dv_dx - du_dy
    return curl.transpose([1,0])



ugos = data_a.ugos.values
vgos = data_a.vgos.values

# np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_adt/ugos',ugos)
# np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_adt/vgos',vgos)

t,at,on = ugos.shape
Curl = np.zeros_like(ugos)
for i in range(t):
    Curl[i,:,:] = MantaCurl2D(ugos[i,:,:],vgos[i,:,:])
    # Curl.append(tmp)
# Curl = np.array(Curl)




# Curl = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_adt/Curl.npy')

CURL = xr.Dataset(
    {
        'curl': (["time","latitude", "longitude"], Curl)#,
        # "mask": (["y","x"],mask)
    },
    coords={
        "longitude": (["longitude"], data_a.longitude),
        "latitude": (["latitude"], data_a.latitude),
        "time": (['time'], data_a.time),
        # "reference_time": pd.Timestamp("2014-09-05"),
    },
)


data_ac = xr.merge([data_a,CURL])

data_ac = data_ac.fillna(-999)
WD = 2*12
data_a_2Y = data_ac.rolling(time=WD,center=True).mean().dropna("time")
data_a_2Y = data_a_2Y.where(data_a_2Y!=-999,drop=False)
data_a_2Y = data_a_2Y.mean(dim='latitude')

# ------ 
data_a_6M = data_ac.rolling(time=6,center=True).mean().dropna("time")
data_a_6M = data_a_6M.where(data_a_6M!=-999,drop=False)
data_a_6M = data_a_6M.mean(dim='latitude')

data_a_6M = data_a_6M.loc[dict(time=slice('1994-01','2018-12'))]
data_a_2Y = data_a_2Y.loc[dict(time=slice('1994-01','2018-12'))]
# data_a_3Y = data_a_3Y.loc[dict(time=slice('1994-01','2018-12'))]

t12 = data_a_2Y.time.values
lon12 = data_a_2Y.longitude.values 
figdata12 = data_a_2Y.curl.values

lon_m12, t_m12 = np.meshgrid(lon12,t12)

# --------

t11 = data_a_6M.time.values
lon11 = data_a_6M.longitude.values 
figdata11 = data_a_6M.curl.values

lon_m11, t_m11 = np.meshgrid(lon11,t11)


# =============================================================================
# 
# =============================================================================
lat10 = data_a_2Y.latitude
lon10 = data_a_2Y.longitude
lon_m10,lat_m10 = np.meshgrid(lon10,lat10)
figdata10 = data_a_2Y.curl.values

def r_vector4cube(x,y,data1,data2,factor):
    xx = x.values.shape[0]
    yy = y.values.shape[0]
    a = np.arange(0,xx,factor[0])
    b = np.arange(0,yy,factor[1])
    
    r_x, r_y = x[a], y[b]
    
    r_data1 = data1.where( (data1.longitude==r_x) & (data1.latitude==r_y), drop=True )
    r_data2 = data2.where( (data2.longitude==r_x) & (data2.latitude==r_y), drop=True )    
    
    return r_x, r_y, r_data1, r_data2 


mask_mlon,mask_Mlon = 130,175
mask_mlat,mask_Mlat = 30, 40


UV_masked = data_a_2Y.where( (data_a_2Y.longitude<mask_mlon) | (data_a_2Y.longitude>mask_Mlon) |\
              (data_a_2Y.latitude<mask_mlat) | (data_a_2Y.latitude>mask_Mlat),drop=False)

lat31 = UV_masked.latitude
lon31 = UV_masked.longitude

figdata32 = UV_masked.ugos
figdata33 = UV_masked.vgos
r_x, r_y, r_data1, r_data2 = r_vector4cube(lon31,lat31,figdata32,figdata33,[3,3])
lon_m32, lat_m32 = np.meshgrid(r_x,r_y)
figdata32 = r_data1
figdata33 = r_data2


# -----------
r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01-01',324,300])
Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

w_path_sig = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/task_adt/vorticity3/'
n = 12
while n < 312:  
    fig, ax = plt.subplots(figsize=(18,11),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
                llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
    # lon_m, lat_m = np.meshgrid(lon_00,lat_00)
    # x, y = m(lon, lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    plt.title('a) Date : '+Sig_set.dates[n] + ' (Vorticity 2Y filtered)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs1 = m.contour(lon_m11,lat_m11,np.flipud(figdata111[n,:,:]),colors='grey',linewidths=2.5,levels=10)
    # plt.clim(-3.3,3.3)
    # plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')

    cs2 = m.pcolormesh(lon_m10,lat_m10,figdata10[n-12,:,:]*10**(6),cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    plt.clim(-3.5,3.5) # plt.clim(-max_figdata02,max_figdata02)
    
    q = m.quiver(lon_m32,lat_m32,figdata32.values[n-12,:,:],figdata33.values[n-12,:,:],
      scale=3.5,headwidth=7.5,headaxislength=10,headlength=13,color='k',
       minlength=1,edgecolor='y',minshaft=1.3,alpha=.7)
    # plt.axis('equal')
    # Unit vector
    # p = plt.quiverkey(q,115.,27,.3,"0.3 m/s",coordinates='data',color='r',
    #                   labelpos='S',alpha=1,labelcolor='w',fontproperties={'size':16},
    #                   labelsep=0.13)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    # h = plt.colorbar(label='',cax=cax);
    h = plt.colorbar(label='$10^{-6}$'+' $[s^{-1}]$',cax=cax);
    # plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(w_path_sig+'Vortisity_2Y'+Sig_set.dates[n])
    plt.tight_layout()
    plt.show()
    n+=1

# =============================================================================
# 
# =============================================================================


t_label= pd.DataFrame({'date':pd.date_range('1994-01-01', periods = 300,freq = 1 * '1m').strftime('%Y')})
lon_label = []
for i in lon11:
    lon_label.append(str(i)[:3]+'°E')

# ---------figure 2Y
fig, ax = plt.subplots(figsize=(8,15),linewidth=1)
plt.pcolormesh(lon_m11, t_m11,figdata11*10**(6),
               cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# plt.colorbar()
plt.clim(-3,3)
plt.axvline(x=lon11[19],color='k',linewidth=3,linestyle='--',alpha=.6)

ytick_location = t11[::12]
ytick_labels = t_label.date.tolist()[::12]
plt.yticks(ticks=ytick_location, labels=ytick_labels, rotation=0, fontsize=18, alpha=.7)
plt.ylabel('$\it{Years}$',fontsize=20,fontweight='light',alpha=.7)
plt.title('b) 6M filtered (23.5$\degree$N)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='bold')

xtick_location = lon11[20::40]
xtick_labels = lon_label[20::40]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=18, alpha=.7)
plt.xlabel('$\it{longitude }$',fontsize=20,fontweight='light',alpha=.7)
plt.grid(axis='y', alpha=.3,linestyle='-.',color='k')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic','alpha':.7})
#label 
h = plt.colorbar(label='$\mathit{10^{-6}}$'+'$[s^{-1}]$',cax=cax);
# plt.savefig(w_path01+'EKEa_2YM',bbox_inches='tight')
plt.tight_layout()
plt.show()

# -----------------
# t_label2= pd.DataFrame({'date':pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y')})
# lon_label2 = []
# for i in lon11:
#     lon_label2.append(str(i)[:3]+'°E')
    
    
fig, ax = plt.subplots(figsize=(8,15),linewidth=1)
plt.pcolormesh(lon_m12, t_m12,figdata12*10**(6),
               cmap=plt.cm.get_cmap('bwr'),shading='gouraud')
# plt.colorbar()
plt.clim(-3.,3.)
plt.axvline(x=lon11[19],color='k',linewidth=3,linestyle='--',alpha=.6)

# ytick_location = t11[::12]
# ytick_labels = t_label.date.tolist()[::12]
plt.yticks(ticks=ytick_location, labels=ytick_labels, rotation=0, fontsize=18, alpha=.7)
plt.ylabel('$\it{Years}$',fontsize=20,fontweight='light',alpha=.7)
plt.title('c) 2Y filtered (23.5$\degree$N)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='bold')

xtick_location = lon12[20::40]
xtick_labels = lon_label[20::40]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=18, alpha=.7)
plt.xlabel('$\it{longitude }$',fontsize=20,fontweight='light',alpha=.7)
plt.grid(axis='y', alpha=.3,linestyle='-.',color='k')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic','alpha':.7})
#label 
h = plt.colorbar(label='$\mathit{10^{-6}}$'+'$[s^{-1}]$',cax=cax);
# plt.savefig(w_path01+'EKEa_2YM',bbox_inches='tight')
plt.tight_layout()
plt.show()




