#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:12:06 2020

@author: shamu
"""

def read_mean4nc2(r_path):
    import os
    import numpy as np
    import xarray as xr
    from tqdm import tqdm
    import os 
    import shutil
    
    nc_list = np.sort([file for file in os.listdir(r_path) if file.endswith(".nc")])
    tmp_len1 = len(nc_list)
    
    os.mkdir(r_path+'tmp_data')
    os.chdir(r_path+'tmp_data')
    for i in nc_list:
        tmp_data = xr.open_dataset(r_path+i,decode_times=True)
        tmp_name = str(tmp_data.time.values[0])[:7]
        
        if not(any(tmp_i==tmp_name for tmp_i in os.listdir())):
            os.mkdir(tmp_name)
            # os.chdir(r_path+'tmp_data/'+tmp_name)
        shutil.move(r_path+i,r_path+'tmp_data/'+tmp_name)
        
    os.mkdir('../Monthly')
    for j in np.sort(os.listdir()):
        print('Mean...'+j)
        tmp_dataset = xr.open_mfdataset(j+'/*.nc', parallel=True)
        Data_Mmean = tmp_dataset.resample(time="1MS").mean(dim="time")
        if j == 'Monthly':
            Data_Mmean.to_netcdf(path='../Monthly/Total.nc',mode='w')
            continue
        Data_Mmean.to_netcdf(path='../Monthly/'+j+'.nc',mode='w')
        tmp_list = os.listdir('./'+j)
        for tmp_j in tmp_list:
            shutil.move(r_path+'tmp_data/'+f"{j}/{tmp_j}",r_path)
    os.rmdir('./'+j)
    os.chdir(r_path)
    shutil.rmtree('./tmp_data')
    tmp_len2 = len(np.sort([file for file in os.listdir(r_path) if file.endswith(".nc")]))
    if tmp_len1+1 != tmp_len2:
        print('File deleted!!!!')
    else :
        print('Check ./Monthly/~')
            


    













