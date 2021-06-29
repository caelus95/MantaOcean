#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:51:32 2021

@author: caelus
"""


import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm

# Difine variables
llcrnrlat = 0 # int  
urcrnrlat = 60 # int
llcrnrlon = 112 # int
urcrnrlon = 260 # inta
var_name = 'adt' # str
r_path = '/home/caelus/dock_2/psi36/DATA/ncfile/CDS/nc/'# str
w_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/'+var_name+\
    '_'+str(llcrnrlat)+'_'+str(urcrnrlat)+'_'+str(llcrnrlon)+'_'+str(urcrnrlon)+'_M.nc'

dir_list = os.listdir(r_path)
nc_list = np.sort([file for file in dir_list if file.endswith(".nc")])

dataset = Dataset(r_path+nc_list[0])

ELat = dataset.variables['latitude'][:]
ELon = dataset.variables['longitude'][:]

lat_co = np.where((ELat >= llcrnrlat) & (ELat <= urcrnrlat))[0]
lon_co = np.where((ELon >= llcrnrlon) & (ELon <= urcrnrlon))[0]
mask = dataset.variables[var_name][0, lat_co, lon_co].mask

lat_rgnl = ELat[lat_co]
lon_rgnl = ELon[lon_co]

Data,n = np.zeros((len(nc_list), len(lat_co), len(lon_co))),0
for i in tqdm(nc_list):
    tmp_data = Dataset(r_path+i)
    Data[n, :, :] = tmp_data.variables[var_name][0, lat_co, lon_co]#.data[0]
    n+=1

Data[Data<-10] = np.nan


ds = xr.Dataset(
    {
        var_name: (["time","lat", "lon"], Data)#,
        # "mask": (["y","x"],mask)
    },
    coords={
        "lon": lon_rgnl,
        "lat": lat_rgnl,
        "time": pd.date_range("1993-01-01", periods=Data.shape[0]),
        # "reference_time": pd.Timestamp("2014-09-05"),
    },
)

# ds.to_masked_Dataset()

Data_Mmean = ds[var_name].resample(time="1MS").mean(dim="time")

# Data_Mmean['lon'] = (["x"], lon_rgnl)
# Data_Mmean['lat'] = (["y"], lat_rgnl)
# Data_Mmean['mask'] = (["y","x"],mask)

Data_Mmean.to_netcdf(path=w_path1,mode='w')

# ds.to_netcdf(path=w_path2,mode='w')

# =============================================================================
# Second method ==> slower 
# =============================================================================
# Data = dataset.variables[var_name][0, lat_co, lon_co]
# mask = Data.mask 
# # nc_list = nc_list[:10] # testing
# for i in tqdm(nc_list[1:]) :
#     tmp_data = Dataset(i)
#     Data = np.dstack([Data,tmp_data.variables[var_name][0, lat_co, lon_co]])
    
# Data = np.transpose(Data,[2,0,1])
# Data.mask = mask

# =============================================================================
# Third
# =============================================================================

# import os
# import glob
# import xarray as xr
# from datetime import datetime
# import numpy as np

# # List all matching files
# # files = glob.glob('/g/data/r78/mc9153/tide_otps/L3_2008_nc3/*.L3m')
# r_path = '/home/shamu/mangrove2/psi36/DATA/ncfile/CDS/nc/'# str
# files = np.sort([file for file in os.listdir(r_path) if file.endswith(".nc")])
# # Create list for 
# individual_files = []

# # Loop through each file in the list
# for i in files:
    
#     # Load a single dataset
#     timestep_ds = xr.open_dataset(i)
#     print(3)
#     # Create a new variable called 'time' from the `time_coverage_start` field, and 
#     # convert the string to a datetime object so xarray knows it is time data
#     timestep_ds['time'] = datetime.strptime(timestep_ds.time_coverage_start, 
#                                            "%Y-%m-%dT%H:%M:%SZ")
#     print(2)
#     # Add the dataset to the list
#     individual_files.append(timestep_ds)
#     print(1)
# # Combine individual datasets into a single xarray along the 'time' dimension
# modis_ds = xr.concat(individual_files, dim='time')


# print(modis_ds)





