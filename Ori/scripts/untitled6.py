#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 04:50:17 2021

@author: caelus
"""
import datetime
import xarray as xr
import os 
import numpy as np
import datetime as dt
import matplotlib.dates as mdt
import pandas as pd
from tqdm import tqdm

r_path = '/home/caelus/dock_1/Working_hub/test_dep/room_4/data/adt_wNpacific_M.nc'

data = xr.open_dataset(r_path,engine='netcdf4')

# date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')


t,at,on = data.adt.shape
Trend = np.zeros([at,on])


t_1 = np.datetime64(datetime.datetime(2013, 1, 1))
t_2 = np.datetime64(datetime.datetime(2015, 9, 1))
    

mm,nn=0,0
for i in tqdm(data.lat):
    for j in data.lon:
        data_s = data.where( (data.time>=t_1)&(data.time<=t_2)&(data.lat==i)&(data.lon==j),drop=True).squeeze()
        
        tmp_t = len(data_s.time.values)
        
        fp1 = np.polyfit(np.arange(tmp_t), data_s.adt.values, 1)
        
        Trend[mm,nn] = fp1[0]
        nn +=1
        print(nn)
    mm+=1
    nn=0
        

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Validations/figs/trend/Trend_p_13_15.npy',Trend)

import matplotlib.pyplot as plt
        
    
plt.pcolormesh(Trend,cmap=plt.cm.get_cmap('seismic'),shading='gouraud')
        
