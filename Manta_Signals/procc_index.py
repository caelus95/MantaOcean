#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:26:48 2021

@author: caelus
"""

# r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'


def sig_pro(r_path,ref_time,Standard=True,WY=2):
    '''
    r_path1 : str
    ref_time : list ex) ['1993-01-01',324,300]
    
    '''
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
    import os 
    
        
    Sig_set_list = np.sort([file for file in os.listdir(r_path) if file.endswith(".npy")])
    
    Sig_sets = pd.DataFrame({},index = pd.date_range(ref_time[0], periods = ref_time[1],freq = 1 * '1m').strftime('%Y-%m'))
    
    for i in Sig_set_list:
        tmp = np.load(r_path+i).reshape(-1)
        columns_name = i.split('.')[0]
        if len(tmp) == ref_time[1] :
            Sig_sets[columns_name] = tmp
        elif len(tmp) == ref_time[2] :
            tmp_Sig_set = pd.DataFrame({columns_name:tmp},index = pd.date_range('1994-01-01', periods = 300,freq = 1 * '1m'))
            Sig_sets = pd.concat([Sig_sets,tmp_Sig_set],axis=1)
        else :
            print(i)
            print('ErroR!!!'*5)
            break
    
  # Sig_set
    # Sig_sets.rename(columns={'EKE_qiu_10_30_120_250':'EKE_qiu','EKE_qiu_10_30_120_250_ceemd_imf5':'EKE_qiu_ceemd_imf5',
    #                      'EKE_qiu_10_30_120_250_pc1':'EKE_qiu_pc1'},inplace=True)  
    
    if Standard :
        Sig_sets = (Sig_sets -Sig_sets.mean())/(Sig_sets.std())


    # Annual mean 
    Sig_sets['dates'] = pd.to_datetime(Sig_sets.index)
    Annual_mean = Sig_sets.groupby(Sig_sets.dates.dt.year).mean()

    # Creating RM (Dataframe)
    RM = Sig_sets.rolling(window=int(12*WY),center=True).mean()
    
    # Removing RM ceemd
    for i in RM.columns:
        if 'ceemd' in i:
            RM.drop(i,axis=1,inplace=True)
        else :
            RM.rename(columns={i:i+'_'+str(WY)+'Y_Rm'},inplace=True)
    
    # Appending name"_Rm"

    Sig_sets = pd.concat([Sig_sets,RM],axis=1)
    
    # Corr Matrix
    Corr_Matrix = Sig_sets[12:-12].corr()

    print('!!!!!!!!!!!!!!!!!!!\nCorrcoef --> 1994~\n!!!!!!!!!!!!!!!!!!!')
    


    
    
    return Sig_sets, Corr_Matrix, Annual_mean
    
    # Sig_set['date'] =  pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')





# lonlat_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/data/qiu_2013/'
# EKE_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/data/'
# data_ = np.load(EKE_path+'EKE_10_30_120_250.npy')
# dataset = data_[:]
# tmp_lonlat = io.loadmat(lonlat_path+'eof.mat')
# lon = tmp_lonlat['lon'][:].reshape(-1)
# lat = tmp_lonlat['lat'][:].reshape(-1)
# sig1 = Sig_set.first_5_2Y_Rm.values
# Slicing_date = ['1993-01','2018-12']


def Spatial_corr(dataset,sig,slicing_date):
    '''
    
    Inputs :
        
        dataset --> numpy array (3D , dim : [time,~,~])
        sig --> DataSeries with index
        Slicing_date --> list / Slicing date ex) ['1993-01', '2018-01']
        
    ------
    
    Retrun : 
        Corr_map 
    
    '''
    from tqdm import tqdm
    import pandas as pd
    import numpy 
    
    tt,ii,jj = dataset.shape
    Corr_map = np.zeros([ii,jj])
    for i in tqdm(range(ii)):
        for j in range(jj):
            tmp_dataset_sig = dataset[:,i,j]
            tmp_Corr_set = pd.DataFrame({'tmp_dataset_sig':tmp_dataset_sig},index = sig.date)
            tmp_Corr_set = tmp_Corr_set.rolling(window=12*WY,center=True).mean()
            tmp_Corr_set['sig'] = sig
            tmp = tmp_Corr_set.loc[Slicing_date[0]:Slicing_date[1]]
            Corr_map[i,j] = np.corrcoef(tmp.sig.dropna(),tmp.tmp_dataset_sig.dropna())[1,0]
        # print(i)
    return(Corr_map)




# Sig_set1 = Sig_set.first_5_2Y_Rm
# Sig_set2 = Sig_set.EKE_qiu_2Y_Rm


def running_corr(sig1,sig2,WY=2):
    
    '''
    Inputs :
        sig1,sig2 --> DataSeries with no index
        
    -------
    Outputs :
        Total Corr
        Corr_vector --> running_corr
    
    '''
    import pandas as pd
    import numpy as np

    Total_corr = np.corrcoef(sig1.dropna(),sig2.dropna())

    Corr_vector = []
    factor = int(WY*12/2)
        
    for i in range(int(factor),len(sig1)-factor):
        # print(i)
        Corr_vector.append(np.corrcoef(sig1[i-factor:i+factor],sig2[i-factor:i+factor])[0,1])
        # print(Corr_vector[i-factor],sig1.date[i])

    return Corr_vector, Total_corr




def linearRegress4Cube(sig,dataset,Slicing_date,method=1):
    '''
    Inputs :
        sig --> DataSeries with index
        dataset --> numpy array (3D [t,~,~])
        Slicing_date --> list / Slicing date ex) ['1993-01', '2018-01']

    -------
    Outputs :
        Coef : 
        p_values :
            
    '''
    import pandas as pd
    from tqdm import tqdm
    import numpy as np
    from scipy import io
    import xarray as xr
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression
    # from sklearn.datasets import fetch_20newsgroups_vectorized
    # from sklearn.feature_selection import chi2


    _,idx,idy = dataset.shape
    Coef,Intercept, p_values = np.zeros([idx,idy]),np.zeros([idx,idy]),np.zeros([idx,idy])
    for i in tqdm(range(idx)):
        for j in range(idy):
            if np.isnan(dataset[:,i,j].mean()):
                # print('dsadsadsadsadsadas')
                Coef[i,j] = np.nan
                Intercept[i,j] = np.nan
                continue

            tmp_dataset_sig = dataset[:,i,j]
            tmp_set = pd.DataFrame({'tmp_dataset_sig':tmp_dataset_sig,'sig':sig},index = sig.index)
            tmp = tmp_set.loc[Slicing_date[0]:Slicing_date[1]]
            X, X_data = tmp.sig.values.reshape(-1,1), tmp.tmp_dataset_sig.values.reshape(-1,1)
     
            if method == 'sm' :
                mod = sm.OLS(X_data,X)
                fii = mod.fit()
                Coef[i,j] = fii.summary2().tables[1]['Coef.'].x1        # print(line_fitter.coef_[1])
                p_values[i,j] = fii.summary2().tables[1]['P>|t|']
            
            else:
                line_fitter = LinearRegression()
                line_fitter.fit(X,X_data)
                Coef[i,j] = line_fitter.coef_
                Intercept[i,j] = line_fitter.intercept_
                
    return Coef, p_values














