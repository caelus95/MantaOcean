plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True

tmp_MEIP = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm >= 0)
tmp_MEIN = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm < 0)

w_path_sig = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/Figures/SUM/'
n = 12
while n < 312:  
    plt.figure(1,figsize=(16,6),dpi=80)
    ax = plt.gca()
    # plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
    #           fontweight='bold',fontsize='48')
    
    # plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
    # Draw Plot
    plt.title('a) Date : '+Sig_set.dates[n] + ' (2Y running mean)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
    plt.plot(Sig_set.dates,Sig_set.ADT_index_2Y_Rm, label='YS index (Yan & Sun 2015)',color='k',linewidth=3,zorder=10)
    plt.plot(Sig_set.dates,Sig_set.EKE_qiu_10_30_120_250_2Y_Rm, label='EKE (Qiu 2013)',color='darkred',linewidth=3,zorder=9)
    plt.plot(Sig_set.dates,Sig_set.PDO_2Y_Rm, label='PDO ',color='darkblue',linewidth=2.5,zorder=8)
    plt.plot(Sig_set.dates,Sig_set.WP_2Y_Rm, label='WP ',color='green',linewidth=2.5,zorder=7)
    
    # Scatter
    
    plt.scatter(Sig_set.dates[n],Sig_set.ADT_index_2Y_Rm[n],
                color='k',marker='o',s=200,zorder=20)
    plt.scatter(Sig_set.dates[n],Sig_set.EKE_qiu_10_30_120_250_2Y_Rm[n],
                color='darkred',marker='o',s=200,zorder=19)
    plt.scatter(Sig_set.dates[n],Sig_set.PDO_2Y_Rm[n],
                color='darkblue',marker='o',s=200,zorder=18)
    plt.scatter(Sig_set.dates[n],Sig_set.WP_2Y_Rm[n],
                color='green',marker='o',s=200,zorder=17)
    
    
    plt.fill_between(Sig_set.dates, tmp_MEIP, color="lightpink",
                      alpha=0.5, label='El-nino',zorder=0)
    plt.fill_between(Sig_set.dates, tmp_MEIN, color="skyblue",
                      alpha=0.5, label='La nina',zorder=1)
    
    plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)
    plt.axvline(x=Sig_set.dates[n],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=0)
    
    # Decoration
    # plt.ylim(50,750)
    xtick_location = Sig_set.dates.tolist()[::12*2]
    xtick_labels = Sig_set.dates.tolist()[::12*2]
    plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
    # plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
    plt.yticks(fontsize=18, alpha=.7)
    # Lighten borders
    plt.gca().spines["top"].set_alpha(.0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(.0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.legend(loc='lower right',fontsize=10)
    plt.grid(axis='y', alpha=.6)
    plt.tight_layout()
    plt.savefig(w_path_sig+'index/index_'+Sig_set.dates[n])
    plt.show()
    n+=1


# ------------ fig WSC ----------------

    
    fig, ax = plt.subplots(figsize=(16,8.5),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=minlat,urcrnrlat=maxlat,\
                llcrnrlon=minlon,urcrnrlon=maxlon,resolution='i')
    # lon_m, lat_m = np.meshgrid(lon_00,lat_00)
    # x, y = m(lon, lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=22,fontweight='bold',color='grey')
    plt.title('b) Date : '+Sig_set.dates[n] + ' (WSCa & Pressa 2Y filtered)', fontproperties='',loc='left',pad=15,fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs = m.pcolormesh(lon_m,lat_m,data)
    # cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    cs1 = m.contour(lon_m10,lat_m10,np.flipud(figdata11[n-12,:,:]),colors='grey',linewidths=2.5,levels=10)
    plt.clim(-3.3,3.3)
    plt.clabel(cs1,fontsize=10,fmt='%1.1f',colors='k')
    
    cs2 = m.pcolormesh(lon_m10,lat_m10,np.flipud(figdata12[n-12,:,:])*10**7,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.clim(-max_figdata02,max_figdata02)
    plt.clim(-.3,.3)
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
    h = plt.colorbar(label='$\mathit{10^{-7} N \cdot m^{-3}}$',cax=cax);
    # plt.savefig(w_path01+'Climatology_WSC_Press',bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(w_path_sig+'/WSCa/WCSa_'+Sig_set.dates[n])
    plt.show()

# ------------ fig ADT ----------------
    
    fig, ax = plt.subplots(figsize=(16,7),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=0,urcrnrlat=60,\
                llcrnrlon=112,urcrnrlon=260,resolution='i')
    # x, y = m(lon, lat)
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,10.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    plt.title('c) Date : '+Sig_set.dates[n] + ' (ADTa 2Y filtered)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs = m.pcolormesh(lon_m,lat_m,data)
    # cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    
    m.plot([120,180,180,120,120],[19,19,27,27,19],color='k',linestyle='--',linewidth=4,alpha=.8)
    
    cs2 = m.pcolormesh(lon_m30,lat_m30,figdata30[n-12,:,:]*10,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.clim(-max_figdata02,max_figdata02)
    plt.clim(-1.5,1.5)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='10 [factor]',cax=cax);
    plt.savefig(w_path_sig+'/ADTa/ADTa_'+Sig_set.dates[n])
    plt.tight_layout()
    plt.show()


# ------------ fig EKE ----------------

    
    fig, ax = plt.subplots(figsize=(16,3),linewidth=1)
    ax = plt.gca()
    m = Basemap(projection='cyl',llcrnrlat=19,urcrnrlat=27,\
                llcrnrlon=120,urcrnrlon=180,resolution='i')
    m.fillcontinents(color='black',lake_color='black')
    m.drawcoastlines()
    m.drawparallels(np.arange(-80.,81.,5.),labels=[True,False,False,False],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    m.drawmeridians(np.arange(-180.,181.,10.),labels=[False,False,False,True],
                    dashes=[2,2],fontsize=20,fontweight='bold',color='grey')
    plt.title('d) Date : '+Sig_set.dates[n] + ' (EKEa 2Y filtered)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
    #plt.suptitle(' UV (mean flow) & speed (anomaly) ',fontstyle='italic',position=(0.5, .92),fontsize=20)
    # cs = m.pcolormesh(lon_m,lat_m,data)
    # cs = m.pcolormesh(lon_m,lat_m,data,cmap=plt.cm.get_cmap('jet'),shading='gouraud')
    
    cs2 = m.pcolormesh(lon_m20,lat_m20,figdata20[n-12,:,:]*100,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
    # plt.clim(-max_figdata02,max_figdata02)
    plt.clim(-2,2)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)
    cax.tick_params(labelsize=15)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    #label 
    h = plt.colorbar(label='100 [factor]',cax=cax);
    plt.savefig(w_path_sig+'/EKEa/EKEa_'+Sig_set.dates[n])
    plt.tight_layout()
    plt.show()
    
    n+=1









