# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 02:22:16 2021

@author: shamu
"""

import xarray as xr
import os 
import numpy as np
import datetime as dt
import matplotlib.dates as mdt
import pandas as pd
from tqdm import tqdm

f_path = '/home/caelus/dock_1/Working_hub/ROMS_dep/task_2/nc/'
w_path = '/home/caelus/dock_1/Working_hub/ROMS_dep/task_2/data/'
data_name ='temp'


# time_starts = mdt.date2num(dt.datetime(1992,1,1))



nc_list = np.sort([file for file in os.listdir(f_path) if file.endswith(".nc")])

# A = xr.open_dataset(f_path+nc_list[0],decode_times=False)
# B = xr.open_dataset(f_path+nc_list[1],decode_times=False)
# C = xr.concat([A,B], dim='ocean_time')

sample = xr.open_dataset(f_path+nc_list[0],decode_times=False)

i=0
while i < len(nc_list):
    if i == 0:
        data = xr.open_dataset(f_path+nc_list[i],decode_times=False)
        data = data[data_name][:,-1,:,:].squeeze() 
        i+=1
        print(i)
    else :
        temp_data = xr.open_dataset(f_path+nc_list[i],decode_times=False)
        data_ = temp_data[data_name][:,-1,:,:].squeeze()
        data = xr.concat([data,data_],dim='ocean_time')
        print(i)
        i+=1

t = pd.date_range("1997-12-05", periods=len(data.ocean_time.values),freq='5D')
da = xr.Dataset(
    {
        data_name: (["time","lat", "lon"], data.values),
    },
    coords={
        "time": t,
        "lat": data.lat_rho[:,0].values,
        "lon": data.lon_rho[0,:].values,
        "reference_time": pd.Timestamp("1992-01-01"),
    },
)
# da = xr.Dataset(
#     {
#         data_name: (["time","x", "y"], data.values),
#     },
#     coords={
#         "time": t,
#         "lat": (["x", "y"], data.lat_rho),
#         "lon": (["x", "y"], data.lon_rho),
#         "reference_time": pd.Timestamp("1992-01-01"),
#     },
# )


Data_Mmean = da.resample(time="1MS").mean(dim="time")
Data_Mmean.to_netcdf(w_path+data_name+'Mm.nc')

Data_Ymean = da.resample(time="1YS").mean(dim="time")
Data_Ymean.to_netcdf(w_path+data_name+'Ym.nc')


# =============================================================================
#  
# =============================================================================

# da = xr.DataArray(
#      {'data':data.values},
#     [
#         ("time", pd.date_range("1992-01-01", periods=len(data.ocean_time.values),freq='5D')),
#         ("lon_rho", data.eta_rho.values),("lat_rho", data.xi_rho.values)
#     ],
# )
'''






K = len(data.ocean_time.values)
data.assign_coords(time =pd.date_range("1992-01-01", periods=K,freq='5D'))

time =pd.date_range("1992-01-01", periods=K,freq='5D')
data.swap_dims({'ocean_time' : 'time'})

ds2 = ds.swap_dims({'Delay': 'Delay_corr'})




t = data.ocean_time.values
for j in t:
    re_t = j/24/3600
    Model_time = mdt.num2date(time_starts+re_t)
    t_index = mdt.num2date(time_starts+re_t)#.strftime('%Y-%m-%d %M:%S')
    print(t_index)


t = A.ocean_time


for i in t:
    re_t = i/24/3600
    Model_time = mdt.num2date(time_starts+re_t)
    t_index = mdt.num2date(time_starts+re_t).strftime('%Y-%m-%d %M:%S')
    
    
    
    
    
    xr.Dataset({"time": datetime.datetime(2000, 1, 1)})
    
    pd.date_range("1992-01-01", periods=len(data.ocean_time.values),freq='5D')
    
    '''
    










