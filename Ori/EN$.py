#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:56:15 2021

@author: caelus
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm


r_path = '/home/caelus/dock_2/psi36/DATA/ncfile/EN4/nc/' 


minlon,maxlon = 112,180
minlat,maxlat = 5,27,
mindepth, maxdepth = 0, 1000
    

data = xr.open_mfdataset(r_path+'*.nc', parallel=True)

tmp = data['temperature'].attrs 
data['temperature'] = data.temperature-273.15
data.temperature.attrs = tmp
data.temperature.attrs['units'] = 'Celsius'


data = data.drop({'temperature_uncertainty','salinity_uncertainty','temperature_observation_weights'\
           ,'salinity_observation_weights','time_bnds','depth_bnds'})
data = data.where( (data.lat>=minlat)&(data.lat<=maxlat)&(data.lon>=minlon)&(data.lon<=maxlon),drop=True)

# =============================================================================
# 
# =============================================================================

Pressure = data.depth # db
Temperature = data.temperature # Celsius
Salinity = data.salinity # psu



from gsw import geo_strf_dyn_height, geostrophic_velocity
from tqdm import tqdm

t,d,at,on = Temperature.shape
dyn_height = np.zeros_like(Temperature)

for i in tqdm(range(t)):
    for j in range(at):
        print('j = '+str(j))
        for k in range(on):
            if np.nanmean(Salinity[i,:,j,k].values) != np.nanmean(Salinity[i,:,j,k].values):
                dyn_height[i,:,j,k] = np.nan
            else: 
                dyn_height[i,:,j,k] = geo_strf_dyn_height(Salinity[i,:,j,k].values, Temperature[i,:,j,k].values,Pressure.values,
                                                          p_ref=0, axis=0,interp_method='pchip')
              
np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/dyn_height_STCC',dyn_height)


lon_mesh, lat_mesh = np.meshgrid(data.lon.values,data.lat.values)



from seawater import gpan, gvel

II = []
JJ = []
geo_vel = np.zeros_like(dyn_height)
for i in dyn_height:
    for j in i:
        JJ.append(geostrophic_velocity_1(j, lon_mesh, lat_mesh, p=0, axis=1))
    II.append(JJ)
    JJ=[]
    
geo_vel =np.array(II) 
        

tmp_A = np.flipud(np.squeeze(geo_vel[0,:,10,10]))
tmp_A[tmp_A<-100] = np.nan
tmp_A[tmp_A>100] = np.nan


tmp_B = np.flipud(np.squeeze(dyn_height[0,:,10,10]))

plt.pcolormesh(tmp_A)
plt.colorbar()
plt.pcolormesh(tmp_B)
plt.colorbar()

import matplotlib.pyplot as plt
plt.contourf(data.temperature[0,0,:,:].values)



# =============================================================================
# 
# =============================================================================

'''
t,d,at,on = Temperature.shape
tmp_ = np.zeros([d,at,on])
Press4D = np.zeros_like(Temperature)
for i in range(at):
    for j in range(on):
        tmp_[:,i,j] = Pressure
for i in range(t):
    Press4D[i,:,:,:] = tmp_

'''


'''
tmp = data['temperature'].attrs 
data['temperature'] = data.temperature-273.15
data.temperature.attrs = tmp
data.temperature.attrs['units'] = 'Celsius'
'''

# A = xr.open_dataset(f_path+nc_list[0],decode_times=False)
# B = xr.open_dataset(f_path+nc_list[1],decode_times=False)
# C = xr.concat([A,B], dim='ocean_time')

'''
sample = xr.open_dataset(r_path+nc_list[0],decode_times=False)
i=0
while i < len(nc_list):
    if i == 0:
        data = xr.open_dataset(r_path+nc_list[i],decode_times=True).where( (data.lat>=minlat)&(data.lat<=maxlat)&(data.lon>=minlon)&(data.lon<=maxlon),
                    drop=True)
        data = data.drop({'temperature_uncertainty','salinity_uncertainty','temperature_observation_weights'\
            ,'salinity_observation_weights','time_bnds','depth_bnds'})
        i+=1
        print(i)
    else :
        data_ = xr.open_dataset(r_path+nc_list[i],decode_times=True).where( (data.lat>=minlat)&(data.lat<=maxlat)&(data.lon>=minlon)&(data.lon<=maxlon),
                    drop=True)
        data_ = data_.drop({'temperature_uncertainty','salinity_uncertainty','temperature_observation_weights'\
            ,'salinity_observation_weights','time_bnds','depth_bnds'})
        data = xr.concat([data,data_],dim='time')
        i+=1
        print(i)
'''





















