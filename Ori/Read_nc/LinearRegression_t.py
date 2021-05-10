#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:11:45 2021

@author: caelus
"""

from scipy import io
import xarray as xr
import statsmodels.api as sm

lonlat_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/data/qiu_2013/'
EKE_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/data/'
data_ = np.load(EKE_path+'EKE_10_30_120_250.npy')
dataset1 = data_[:]

tmp_lonlat = io.loadmat(lonlat_path+'eof.mat')
lon = tmp_lonlat['lon'][:].reshape(-1)
lat = tmp_lonlat['lat'][:].reshape(-1)

sig1 = Sig_set.first_5_2Y_Rm.values
Slicing_date = ['1994-1','2005-10']


from sklearn.linear_model import LinearRegression
# from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import chi2

X = sig1

_,idx,idy = data_.shape
Coef1,Intercept1 = np.zeros([idx,idy]),np.zeros([idx,idy])
for i in tqdm(range(idx)):
    for j in range(idy):
        if np.isnan(data_[:,i,j].mean()):
            Coef1[i,j] = np.nan
            Intercept1[i,j] = np.nan
            continue
        
        tmp_dataset_sig = data_[:,i,j]
        tmp_Corr_set = pd.DataFrame({'tmp_dataset_sig':tmp_dataset_sig,'sig1':sig1},index = Sig_set.date)
        tmp = tmp_Corr_set.loc[Slicing_date[0]:Slicing_date[1]]
        X, X_data = tmp.sig1.values.reshape(-1,1), tmp.tmp_dataset_sig.values.reshape(-1,1)
        line_fitter = LinearRegression()
        line_fitter.fit(X,X_data)
        Coef1[i,j] = line_fitter.coef_
        # print(line_fitter.coef_[1])
        Intercept1[i,j] = line_fitter.intercept_
        # print(line_fitter.intercept_[1])
        
        
        # data = fetch_20newsgroups_vectorized()
        # X, y = data.data, data.target
        # scores, pvalues = chi2(X, X_data)
        
# =============================================================================
#         
# =============================================================================
      
X = sig1

_,idx,idy = data_.shape
Coef,p_values = np.zeros([idx,idy]),np.zeros([idx,idy])
for i in tqdm(range(idx)):
    for j in range(idy):
        if np.isnan(data_[:,i,j].mean()):
            Coef[i,j] = np.nan
            p_values[i,j] = np.nan
            continue
        
        tmp_dataset_sig = data_[:,i,j]
        tmp_Corr_set = pd.DataFrame({'tmp_dataset_sig':tmp_dataset_sig,'sig1':sig1},index = Sig_set.date)
        tmp = tmp_Corr_set.loc[Slicing_date[0]:Slicing_date[1]]
        X, X_data = tmp.sig1.values.reshape(-1,1), tmp.tmp_dataset_sig.values
        mod = sm.OLS(X,X_data)
        fii = mod.fit()

        Coef[i,j] = fii.summary2().tables[1]['Coef.'].x1        # print(line_fitter.coef_[1])
        p_values[i,j] = fii.summary2().tables[1]['P>|t|']
        # print(line_fitter.intercept_[1])
        
        
        # data = fetch_20newsgroups_vectorized()
        # X, y = data.data, data.target
        # scores, pvalues = chi2(X, X_data)
          
        
        
        
plt.pcolormesh(Coef,cmap='seismic',shading='gouraud')
plt.colorbar()
plt.clim([-0.05,0.05])

plt.pcolormesh(Coef2,cmap='seismic',shading='gouraud')
plt.colorbar()
plt.clim([-10,1])


plt.pcolormesh(p_values,cmap='seismic',shading='gouraud')
plt.colorbar()
plt.clim([0,0.05])



non_sig_coor = np.where(p_values2>0.05)
# p_values[a] = 0

Coef2[non_sig_coor] = np.nan

plt.plot(tmp.sig1.values.reshape(-1,1))
plt.plot(tmp.tmp_dataset_sig.values.reshape(-1,1))        
        

        

mod = sm.OLS(X,X_data)

fii = mod.fit()
p_values = fii.summary2().tables[1]['P>|t|']
fii.summary2().tables[1]['Coef.'].x1


X2 = sm.add_constant(X)
est = sm.OLS(X,X_data)
est2 = est.fit()
print(est2.summary())


_,idx,idy = data_.shape
Coef2,p_values2 = np.zeros([idx,idy]),np.zeros([idx,idy])
for i in tqdm(range(idx)):
    for j in range(idy):
        if np.isnan(data_[:,i,j].mean()):
            Coef2[i,j] = np.nan
            p_values2[i,j] = np.nan
            continue
        
        tmp_dataset_sig = data_[:,i,j]
        tmp_Corr_set = pd.DataFrame({'tmp_dataset_sig':tmp_dataset_sig,'sig1':sig1},index = Sig_set.date)
        tmp = tmp_Corr_set.loc[Slicing_date[0]:Slicing_date[1]]
        X, X_data = tmp.sig1.values.reshape(-1,1), tmp.tmp_dataset_sig.values
        mod = sm.OLS(X,X_data)
        fii = mod.fit()

        Coef2[i,j] = fii.summary2().tables[1]['Coef.'].x1        # print(line_fitter.coef_[1])
        p_values2[i,j] = fii.summary2().tables[1]['P>|t|']
        # print(line_fitter.intercept_[1])
        



