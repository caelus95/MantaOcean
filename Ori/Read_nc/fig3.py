#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:07:27 2020

@author: shamu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 18:20:13 2020

@author: shamu
"""

import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

r_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/ugos_Npacific_M.nc'
r_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/vgos_Npacific_M.nc'

w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/Sigs/'
# w_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/figs/Yan_Sun_mean_area_3.png'

Data1 = xr.open_dataset(r_path1)
Data2 = xr.open_dataset(r_path2)

ugos,vgos = Data1['ugos'],Data2['vgos']

ugos_a = (ugos - ugos.mean(axis=0))*100
vgos_a = (vgos - vgos.mean(axis=0))*100 

EKE = (ugos_a**2 + vgos_a**2)/2 
# EKE_a = EKE - EKE.mean()

LAT,LON = Data1['lat'], Data1['lon']

mlat1,Mlat1,mlon1,Mlon1 = 22.5, 24, 123, 124

EKE_S = EKE.where( (EKE.lat >= mlat1) & (EKE.lat<=Mlat1) & (EKE.lon >= mlon1) & (EKE.lon <=Mlon1) ,
                   drop=True).mean(axis=(1,2),skipna=True)


date = pd.date_range('1993-01-01', periods = 324, freq = 1 * '1m').strftime('%Y-%m') # or freq = 3 * '1D'


df = pd.DataFrame(EKE_S)
df['date']= date
import pandas as pd

pd_sig = pd.DataFrame(EKE_S)

R_m_sig = pd_sig.rolling(window=12*2,center=True).mean()
plt.plot(R_m_sig)

# 
plt.figure(figsize=(16,10), dpi= 80)
plt.plot(date, EKE_S, color='k',linewidth=3)

xtick_location = df.index.tolist()[::12]
xtick_labels = df.date.tolist()[::12]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=45, fontsize=15, alpha=.7)
# plt.title('IMF05', fontsize=22)
plt.yticks(fontsize=15, alpha=.7)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)

plt.legend(loc='lower right')
plt.grid(axis='both', alpha=.3)
# plt.savefig(w_path1,dpi=150)
plt.show()

np.save(w_path1+'wang_eddybox_2.npy',EKE_S)





# EKE




# Vorticity



