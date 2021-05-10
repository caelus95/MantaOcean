#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 13:04:56 2020

@author: shamu
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/latest/sigs/first_5.npy'
r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Validations/eemd_results/eemd_1000.npy'

# r_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/analysed_sst_Npacific_M.nc'
# w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/latest/figs/'

sig = np.load(r_path)

S = pd.read_csv(r_path)
S.index = S.Date

sig = S.loc[199301:201912]

sig.reset_index(drop=True,inplace=True)
SIG = pd.DataFrame(sig,index = pd.date_range('1993-01-01', periods = 324,
                                                  freq = 1 * '1m')) # .to_period()

# np.save('/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/latest/sigs/PDO',SIG.values)
# =============================================================================
# Remove Seasonality
# =============================================================================

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(SIG, model='additive')

result.plot()
plt.show()

plt.plot(result.trend)
plt.plot(result.seasonal)
plt.plot(result.resid)


seasonality = result.seasonal

Ori_Rm_seasonality = SIG.values.reshape(-1) - seasonality.values

# imf5 = SIG1['5_mod']
imf5 = SIG.iloc[:,0]

plt.plot(imf5)
plt.grid()


plt.plot(imf5)

Data = xr.open_dataset(r_path2)
data = Data['analysed_sst'].data

data[data<-10] = np.nan

# =============================================================================
# Simple running mean
# =============================================================================

import pandas as pd

pd_sig = pd.DataFrame(SIG)

R_m_sig = pd_sig.rolling(window=12*2,center=True).mean()
plt.plot(R_m_sig)


# =============================================================================
# LPF
# =============================================================================

import numpy as np
from scipy.signal import butter, filtfilt 
import matplotlib.pyplot as plt 

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff/nyq
    b,a = butter(order, normal_cutoff, btype='low', analog=False)
    y=filtfilt(b, a, data)
    return y 

Signal = pd_sig.values.reshape(-1)
L = len(Signal) 
T = 6 #period(time interval)
fs = L/T #rate(total number of samples/period)
cutoff = 1 #cutoff frequency

nyq= 0.5*fs
order=2 #filter order
n= int(T*fs) #total number of samples 

LPF = butter_lowpass_filter(Signal, cutoff, fs, order) 

plt.plot(LPF)


# # =============================================================================
# # Pyeemd <-- Untrustworthy
# # =============================================================================
# from PyEMD import EEMD,Visualisation

# # if __name__ == "__main__":
# #     EEMD(trials=500,noise_width=0.2)
# #     eIMFs = eemd(ori_s)

# t = np.arange(324)

# # In general:
# components = EEMD(trials=5000,noise_width=ori_s.std()*0.2)(ori_s)
# imfs, res = components[:-1], components[-1]

# plt.plot(imfs[4,:])

# vis = Visualisation()
# vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
# vis.plot_instant_freq(t, imfs=imfs)
# vis.show()



# =============================================================================
# plot 1D
# =============================================================================
date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True
plt.rcParams['font.size'] = 14


plt.figure(figsize=(16,10), dpi= 80)
plt.plot(date,SIG2.values,color=[.7,.7,.7],linewidth=2,label='original')
plt.plot(date,imf5+.35,label='imf5(ceemdan)',color='k',linewidth=3.5)
plt.plot(date,R_m_sig.values,label='running mean',color='darkred',linewidth=3.5)
plt.plot(date,LPF,label='LPF',color='darkblue',linewidth=3.5)
plt.plot(date,result.trend,label='???',color='darkgreen',linewidth=2.5)



plt.ylim(.25,.65)
xtick_location = date.tolist()[::12]
xtick_labels = date.tolist()[::12]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=45, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper left')
plt.grid(axis='y', alpha=.3)
# plt.savefig(w_path1+'decompo')
plt.show()

# =============================================================================
# Regression
# =============================================================================
from sklearn.linear_model import LinearRegression

X = SIG.values.reshape(-1,1)

_,idx,idy = data.shape
Coef,Intercept = np.zeros([idx,idy]),np.zeros([idx,idy])

for i in tqdm(range(idx)):
    for j in range(idy):
        if np.isnan(data[:,i,j].mean()):
            Coef[i,j] = np.nan
            Intercept[i,j] = np.nan
            continue
        line_fitter = LinearRegression()
        line_fitter.fit(X, data[:,i,j].reshape(-1,1))
        Coef[i,j] = line_fitter.coef_
        Intercept[i,j] = line_fitter.intercept_


from scipy import io
AA = io.loadmat('/home/shamu/mangrove2/psi36/MATLAB_linux/KUROSHIO/Signals/PTO.mat')
pto = AA['PTO']
# y_predicted = line_fitter.predict(X.reshape(-1,1))

'''
import statsmodels.api as sm

mod = sm.OLS(data[:,i,j].reshape(-1,1),X)

fii = mod.fit()

p_values = fii.summary2().tables[1]['P>|t|']


from scipy import stats
tTestResult = stats.ttest_ind(X, data[:,i,j].reshape(-1,1))

tTestResultDiffVar = stats.ttest_ind(X, data[:,i,j].reshape(-1,1), equal_var=False)

from scipy.stats import ttest_ind
tstat, pval = ttest_ind(X, data[:,180,200].reshape(-1,1))
'''

# =============================================================================
# Corr
# =============================================================================
 
Data = Data['adt'].values

T,L,M = Data['adt'].values.shape

corr_map = np.zeros([L,M])
for i in range(L):
    for j in range(M):
        corr_map[i,j] = np.corrcoef(SIG.values.reshape([-1,1]),Data[:,i,j])[1,0]



# =============================================================================
# Composite
# =============================================================================

# imf55 = imf5[240:275]


posi_co = np.where(imf5>0)[0]
nega_co = np.where(imf5<=0)[0]

sla = data - np.nanmean(data,axis=0)

Posi = np.nanmean(sla[posi_co,:,:],axis=0)
Nega = np.nanmean(sla[nega_co,:,:],axis=0)

import pandas as pd
t = pd.date_range('1993-01-01', periods = 324, freq = 1 * '1m') # or freq = 3 * '1D'
plt.plot(t,imf5)
#

# Posi = np.nanmean(sla[posi_co[49:],:,:],axis=0)
Posi = np.nanmean(sla[posi_co[:],:,:],axis=0)

# Nega = np.nanmean(sla[nega_co[:120],:,:],axis=0)
Nega = np.nanmean(sla[nega_co,:,:],axis=0)



# def Pvalue(Vari,Sampling_n,Compo_n):
#     from scipy.stats import norminvgauss
#     import numpy as np
#     from tqdm import tqdm
#     # import random
#     # random.seed('')
#     t,idx,idy = Vari.shape
#     R_Vari = np.zeros([Sampling_n,idx,idy])    

#     x_Mean = np.zeros([idx,idy])
#     x_Var =  np.zeros_like(x_Mean)
#     x_Norminv = np.zeros([2,idx,idy])
#     print('Processing...(1/2)')
#     for i in tqdm(range(Sampling_n)):
#         for j in range(idx):
#             for k in range(idy):
#                 R_num = np.random.randint(low=0,high=t,size=Compo_n)
#                 R_Vari[i,j,k] = np.squeeze( np.nanmean(Vari[R_num,j,k]) )
   
#     print('Processing...(2/2)')
#     for i in range(idx):
#         for j in range(idy):
#             x_Mean[i,j] = R_Vari[:,i,j].mean()
#             x_Var[i,j] = R_Vari[:,i,j].std()
#             x_Norminv[:,i,j] = norminvgauss.ppf([.05,.95],x_Mean[i,j],x_Var[i,j])
#     return x_Mean, x_Var, x_Norminv
    
# _,_,x_Norminv = Pvalue(sla,100,len(posi_co))


# =============================================================================
# plotting (Basemap)
# =============================================================================

import xarray as xr
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

Data = xr.open_dataset(r_path2)
LON,LAT = Data['lon'],Data['lat']

plt.contourf(Nega,cmap='seismic',levels=30)
plt.clim(-.2,.2)
plt.colorbar()

plt.pcolormesh(corr_map,cmap='seismic')
plt.clim(-.5,.5);
plt.colorbar()

plt.contourf(Coef,cmap='seismic',levels=30)
plt.clim(-10.,10.)
plt.colorbar()


# =============================================================================
# 
# =============================================================================
st = 258
ed = 287
imf55 = imf5[st:ed]
data1 = np.nanmean(sla[st:ed,:,:],axis=0) 

# Posi = np.nanmean(data1[posi_co,:,:],axis=0)
# Nega = np.nanmean(data1[nega_co,:,:],axis=0)



Minlat,Maxlat,Minlon,Maxlon = 20,45,115,150

plt.figure(figsize=(14, 12))
plt.title('Regressed SST',fontsize=38,fontweight='bold')
ax = plt.gca()
m = Basemap(projection='cyl',llcrnrlat=Minlat,urcrnrlat=Maxlat,
            llcrnrlon=Minlon,urcrnrlon=Maxlon,resolution='i')
m.drawmapboundary(linewidth=3)
lon2, lat2 = np.meshgrid(LON,LAT)
m.fillcontinents(color='black',lake_color='black')
m.drawcoastlines(),
m.drawparallels(np.arange(-10,60,5),labels=[True,False,False,False],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
m.drawmeridians(np.arange(120.,262.,10.),labels=[False,False,False,True],
                dashes=[2,2],fontsize=24,fontweight='bold',color='grey')
# plt.title(' Positive phase (sla)', position=(.5, 5.0+0.1),fontweight='bold',fontsize=46)
# plt.suptitle(' 
x, y = m(LON.values,LAT.values)
m.pcolormesh(x,y,Coef,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
plt.clim(-10,10)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
h = plt.colorbar(label=f'',cax=cax);
# plt.savefig(w_path1+'Regress_ghrsst', dpi=200)
plt.show()









