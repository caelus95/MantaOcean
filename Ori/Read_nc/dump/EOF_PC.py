#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:57:34 2021

@author: caelus
"""

ugos = xr.open_dataset('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ugos_20_28_120_135_M.nc')
vgos = xr.open_dataset('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/vgos_20_28_120_135_M.nc')

Speed = (ugos.ugos**2+vgos.vgos**2)**(1/2)


Time_1 = ['2005-11','2011-01'] 
Time_2 = ['1993-01','2005-10']

data_s1 = Speed.loc[dict(time=slice(Time_1[0],Time_1[1]))]
data_s2 = Speed.loc[dict(time=slice(Time_2[0],Time_2[1]))]

s1 = data_s1.values
lon1 = Speed.lon
lat1 = Speed.lat

s2 = data_s2.values
lon2 = data_s2.lon
lat2 = data_s2.lat



np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/s_0511',s1)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/s_9305',s2)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/lon1',lon)
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/lat1',lat)





eof_0511 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/eof_0511.npy')
eof_9305 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/eof_9305.npy')
pc_0511 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/pc_0511.npy')
pc_9305 = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/tmp/pc_9305.npy')



# -------------------- figure -----------------------
t = pd.date_range('1993-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')



ii = 0

fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=20,urcrnrlat=28,\
            llcrnrlon=120,urcrnrlon=135,resolution='i')
lon_m, lat_m = np.meshgrid(lon1,lat1)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,2.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,2.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
plt.title(' EOF 1993-01 ~ 2005-10 ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)

cs2 = m.pcolormesh(lon_m,lat_m,eof_9305[ii,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-1,1)
# cs2 = m.pcolormesh(lon_m,lat_m,mapdata_2,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(min_mapdata_2,max_mapdata_2)
# m.quiver(lon_mr,lat_mr,data_mur,data_mvr,headwidth=5,headaxislength=10,headlength=10)
#m.quiver(Acp,Bcp,u_s,v_s,scale=7,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax);
h = plt.colorbar(label='$\mathit{10 [m]}$',cax=cax);
# plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
plt.tight_layout()
plt.show()


# --------------------------------


plt.figure(1)
plt.plot(t[:154],pc_9305[ii,:])
plt.title('PC'+ str(ii)+' 1993-01 ~ 2005-10')
xtick_location = t[:154].tolist()[::12*2]
xtick_labels = t[:154].strftime('%Y-%m').tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=30, fontsize=16, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=16, alpha=.7,fontweight='bold')
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='x', alpha=.3)
plt.show()

### ----------------------------

fig, ax = plt.subplots(figsize=(15,8),linewidth=1)
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=20,urcrnrlat=28,\
            llcrnrlon=120,urcrnrlon=135,resolution='i')
lon_m, lat_m = np.meshgrid(lon1,lat1)
# x, y = m(lon, lat)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,2.),labels=[True,False,False,False],
                dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
m.drawmeridians(np.arange(-180.,181.,2.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
plt.title(' EOF 2005-11 ~ 2011-01 ', fontproperties='', position=(0.5, 1.0+0.07), fontsize=40,fontweight='bold')
#plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
# cs = m.pcolormesh(lon_m,lat_m,data)

cs2 = m.pcolormesh(lon_m,lat_m,eof_0511[ii,:,:]*-10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(-max_figdata02,max_figdata02)
plt.clim(-1,1)
# cs2 = m.pcolormesh(lon_m,lat_m,mapdata_2,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
# plt.clim(min_mapdata_2,max_mapdata_2)
# m.quiver(lon_mr,lat_mr,data_mur,data_mvr,headwidth=5,headaxislength=10,headlength=10)
#m.quiver(Acp,Bcp,u_s,v_s,scale=7,headwidth=5,headaxislength=10,headlength=10,color=[.25,.25,.25])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
#label 
h = plt.colorbar(label='',cax=cax)
h = plt.colorbar(label='$\mathit{-10m}$',cax=cax)
# plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
plt.tight_layout()
plt.show()


plt.figure(2)
plt.plot(t[154:154+63],pc_0511[ii,:])
plt.title('PC'+ str(ii)+' 2005-11 ~ 2011-01')
xtick_location = t[154:154+63].tolist()[::12*1]
xtick_labels = t[154:154+63].tolist()[::12*1]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=30, fontsize=16, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=16, alpha=.7,fontweight='bold')
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='x', alpha=.3)
plt.show()



t[154:154+63],pc_0511[0,:]

tmp_sig1 = pd.DataFrame({'pc0511':pc_0511[0,:]},index=t[154:154+63])
tmp_sig2 = pd.DataFrame({'pc9305':pc_9305[0,:]},index=t[:154])



r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

Sig_set,Corr_map,Annual_mean = sig_pro(r_path4,['1993-01',324,300],Standard=True)

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')


Sig_set = pd.concat([Sig_set,tmp_sig1,tmp_sig2],axis=1)

# ------------ test fig ----------------

plt.figure(1,figsize=(16,6),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.title('a) EOF PC time series', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
# plt.plot(Sig_set.dates,Sig_set.ADT_index_2Y_Rm, label='YS index (Yan & Sun 2015)',color='k',linewidth=3,zorder=10)
# plt.plot(Sig_set.dates,Sig_set.EKE_qiu_10_30_120_250_2Y_Rm, label='EKE (Qiu 2013)',color='darkred',linewidth=3,zorder=9)
# plt.plot(Sig_set.dates,Sig_set.PDO_2Y_Rm, label='PDO ',color='darkblue',linewidth=2.5,zorder=8)
plt.plot(Sig_set.dates[:154+62],Sig_set.pc9305[:154+62], label='pc1 1993-01~2005-10',color='r',linewidth=2.5,zorder=3)
plt.plot(Sig_set.dates[:154+62],Sig_set.ADT_index[:154+62], label='KVTe index ',color='grey',linewidth=5,zorder=1)
plt.plot(Sig_set.dates[:154+62],Sig_set.ADT_index_2Y_Rm[:154+62], label='KVTe index (2Y Rm)',color='k',linewidth=3,zorder=2)
plt.plot(Sig_set.dates[:154+62],-Sig_set.pc0511[:154+62], label='-pc1 2005-11~2011-01',color='b',linewidth=2.5,zorder=3)
# plt.plot(Sig_set.dates,Sig_set.WP_2Y_Rm, label='WP ',color='green',linewidth=2.5,zorder=7)

# Scatter

# plt.scatter(Sig_set.dates[n],Sig_set.ADT_index_2Y_Rm[n],
#             color='k',marker='o',s=200,zorder=20)
# plt.scatter(Sig_set.dates[n],Sig_set.EKE_qiu_10_30_120_250_2Y_Rm[n],
#             color='darkred',marker='o',s=200,zorder=19)
# plt.scatter(Sig_set.dates[n],Sig_set.PDO_2Y_Rm[n],
#             color='darkblue',marker='o',s=200,zorder=18)
# plt.scatter(Sig_set.dates[n],Sig_set.WP_2Y_Rm[n],
#             color='green',marker='o',s=200,zorder=17)


# plt.fill_between(Sig_set.dates, tmp_MEIP, color="lightpink",
#                   alpha=0.5, label='El-nino',zorder=0)
# plt.fill_between(Sig_set.dates, tmp_MEIN, color="skyblue",
#                   alpha=0.5, label='La nina',zorder=1)

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)
# plt.axvline(x=Sig_set.dates[n],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=0)

# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.dates[:154+62].tolist()[::12*2]
xtick_labels = Sig_set.dates[:154+62].tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right',fontsize=10)
plt.grid(axis='y', alpha=.6)
plt.tight_layout()
# plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
plt.show()
































