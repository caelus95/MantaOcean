import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from scipy import io


r_path1 = '/home/shamu/mangrove2/psi36/MATLAB_linux/KUROSHIO/Signals/YanSun_dif.mat'
r_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/Sigs/Yan_Sun_dif_area_1.npy'
r_path3 = '/home/shamu/Wormhole/tmp/ys_eemd_11.npy'
r_path4 = '/home/shamu/Wormhole/tmp/ys_eemd_20.npy'
r_path5 = '/home/shamu/Wormhole/tmp/ys_ceemdan_11.npy'
r_path6 = '/home/shamu/Wormhole/tmp/ys_ceemdan_20.npy'

r_path7 = '/home/shamu/Wormhole/tmp/bo_qiu/Kuro_ceemdan.npy'
r_path8 = '/home/shamu/Wormhole/tmp/bo_qiu/EKE_bo_qiu_ceemdan.npy'
r_path9 = '/home/shamu/Wormhole/tmp/bo_qiu/xcorr.npy'
r_path10 = '/home/shamu/Wormhole/tmp/bo_qiu/row_boqiu_EKE.npy'

w_path1 = '/home/shamu/Wormhole/tmp/bo_qiu/'
# w_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/figs/Yan_Sun_mean_area_12.png'


SSS = io.loadmat(r_path1)

SS = SSS['YanSun_eemd']

# Sig1 = np.load(r_path2)
# Sig2 = np.load(r_path3)
# Sig3 = np.load(r_path4) 

Sig7 = np.load(r_path7)
Sig8 = np.load(r_path8)
Sig9 = np.load(r_path9)
Sig10 = np.load(r_path10)




date = pd.date_range('1993-01-01', periods = 312,freq = 1 * '1m').strftime('%Y-%m')
date2 = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True
plt.rcParams['font.size'] = 18



plt.figure(figsize=(16,8), dpi= 80)

# plt.plot(date,SS[:,0]-np.mean(SS[:,0]),color=[.7,.7,.7],linewidth=2.5,linestyle='-',label='Ori dif(~2018)')
plt.plot(date2,Sig10 ,color='darkred',linewidth=3,linestyle='-',label=r'EKE (Bo qiu)')
# plt.plot(date2,Sig8/800 ,color='darkblue',linewidth=3,linestyle='-',label=r'EKE (bo qiu) imf5 (ceemdan)')
# plt.plot(date2,Sig3 - .04 ,color='darkblue',linewidth=3,linestyle='-',label=r'imf5 26.5$\degree$N (eemd)')
# plt.plot(date2,Sig4 - .04,color='blue',linewidth=3,linestyle='-.',label=r'imf5 27.25$\degree$N (eemd)')

# plt.plot(date,SS[:,5],color='darkred',linewidth=3,label='imf5(~2018, points)')
# plt.plot(date2,Sig2 - .1,color='darkblue',linewidth=3,label=r'imf5(~2019,Regional mean 27.25$\degree$N)')
# plt.plot(date2,Sig3 - .2,color='k',linewidth=3,label=r'imf5(~2019, Regional mean 26.5$\degree$N)')



# plt.ylim(-.06,.06)
xtick_location = date2.tolist()[::12*2]
xtick_labels = date2.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='best')
plt.grid(axis='y', alpha=.3)
plt.savefig(w_path1+'row_bo_qiu_EKE')
plt.show()





# =============================================================================
# 
# =============================================================================

plt.bar(Sig9.reshape(-1),Sig10.reshape(-1),label='Xcorr',color='k')
plt.xlim(-75,75)
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='best')
plt.grid(axis='y', alpha=.3)
plt.savefig(w_path1+'xcorr')
plt.show()




