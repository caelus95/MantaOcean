#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 18:58:53 2020

@author: shamu
"""


from netCDF4 import Dataset
r_path ='/home/shamu/HUB/eddy_trajectory_2.0exp_19930101_20191015.nc'
A = Dataset(r_path)


import xarray as xr
import numpy as np
# Load dataset with xarray
r_path = '/home/shamu/mangrove2/psi36/DATA/ncfile/MesoscaleEddy/eddy_trajectory_2.0exp_19930101_20191015.nc'
with xr.open_dataset(r_path, decode_cf=False)  as h:
 
    lon_min, lon_max, lat_min, lat_max =1,5, -28, -27
    t =25150# Select a specific eddy with date and area, only some observations
    subset = h.sel(obs=(h.longitude> lon_min) & (h.longitude< lon_max) & (h.latitude> lat_min) & (h.latitude< lat_max) & (h.time==25147))# Extract full track
    subset = h.isel(obs=np.in1d(h.track, subset.track))# Store selected dataprint(subset)
    subset.to_netcdf('output_eddy.nc')
    
    
    
from matplotlib import pyplot as plt
# Load dataset with xarray
ax = plt.subplot(111) 
with xr.open_dataset('output_eddy.nc')as h:
    N =20# plot contour every N
    ax.plot((h.speed_contour_longitude[::N].T + 180) % 360 - 180, h.speed_contour_latitude[::N].T,'r')# plot path
    ax.plot((h.longitude + 180) % 360 - 180, h.latitude,'b', label='eddy path')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid()
plt.show()
 



import xarray as xr
import numpy as np
# Load dataset with xarray
r_path = '/home/shamu/mangrove2/psi36/DATA/ncfile/MesoscaleEddy/eddy_trajectory_2.0exp_19930101_20191015.nc'
Data = xr.open_dataset(r_path, decode_cf=False)



import datetime as dt
from matplotlib.dates import date2num,num2date,datestr2num

Data.time.data

num2date(15706+datestr2num('1950-01-01')) # 1993-01-01 


lon_min, lon_max, lat_min, lat_max =1,5, -28, -27
t =25150# Select a specific eddy with date and area, only some observations
subset = h.sel(obs=(h.longitude> lon_min) & (h.longitude< lon_max) & (h.latitude> lat_min) & (h.latitude< lat_max) & (h.time==25147))# Extract full track
subset = h.isel(obs=np.in1d(h.track, subset.track))# Store selected dataprint(subset)
subset.to_netcdf('output_eddy.nc')



# =============================================================================
# Finding eddies at longitude-latitude-box 
# =============================================================================


from netCDF4 import Dataset
r_path ='/home/shamu/HUB/eddy_trajectory_2.0exp_19930101_20191015.nc'
import xarray as xr
import numpy as np

mlat,Mlat = 10,30 

Data = xr.open_dataset(r_path, decode_cf=False)


eddy_lat,eddy_lon = Data.latitude, Data.longitude

eddy_lat.where((eddy_lat>Mlat)&(eddy_lat<mlat))




# =============================================================================
# 
# =============================================================================


import os
os.chdir('/home/shamu/HUB/')
import xarray as xr
import numpy as np
# Load dataset with xarray
r_path ='/home/shamu/HUB/eddy_trajectory_2.0exp_19930101_20191015.nc'
with xr.open_dataset(r_path, decode_cf=False)  as h:

    lon_min, lon_max, lat_min, lat_max =120,140, 10,20
    t_min,t_max = 15706,18261 # Select a specific eddy with date and area, only some observations
    subset = h.sel(obs=(h.longitude > lon_min) & (h.longitude < lon_max) &
                   (h.latitude> lat_min) & (h.latitude< lat_max) & (h.time>=t_min) & (h.time<=t_max))# Extract full track
    # subset = h.isel(obs=np.in1d(h.track, subset.track))# Store selected dataprint(subset) (slicing)
    subset
    
    

import datetime as dt
from matplotlib.dates import date2num,num2date,datestr2num
num2date(18261+datestr2num('1950-01-01')) # 1993-01-01 
# 15706 == 19930101
    
from matplotlib import pyplot as plt
# Load dataset with xarray
ax = plt.subplot(111) 
with xr.open_dataset('output_eddy.nc')as h:
    N =20# plot contour every N
    ax.plot((h.longitude[::N].T + 180) 
            , h.latitude[::N].T,'r')# plot path
    ax.plot((h.longitude + 180) % 360 - 180, h.latitude,'b', label='eddy path')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid()
plt.show()
 



import xarray as xr
import numpy as np
# Load dataset with xarray
r_path = '/home/shamu/mangrove2/psi36/DATA/ncfile/MesoscaleEddy/eddy_trajectory_2.0exp_19930101_20191015.nc'
Data = xr.open_dataset(r_path, decode_cf=False)


import datetime as dt
from matplotlib.dates import date2num,num2date,datestr2num

Data.time.data

num2date(15706+datestr2num('1950-01-01')) # 1993-01-01 


lon_min, lon_max, lat_min, lat_max =1,5, -28, -27
t =25150# Select a specific eddy with date and area, only some observations
subset = h.sel(obs=(h.longitude> lon_min) & (h.longitude< lon_max) & (h.latitude> lat_min) & (h.latitude< lat_max) & (h.time==25147))# Extract full track
subset = h.isel(obs=np.in1d(h.track, subset.track))# Store selected dataprint(subset)
subset.to_netcdf('output_eddy.nc')





from matplotlib import pyplot as plt
# Load dataset with xarray
ax = plt.subplot(111) 
with xr.open_dataset('output_eddy.nc')as h:
    N =20# plot contour every N
    ax.plot((h.speed_contour_longitude[::N].T + 180) % 360 - 180, h.speed_contour_latitude[::N].T,'r')# plot path
    ax.plot((h.longitude + 180) % 360 - 180, h.latitude,'b', label='eddy path')
    ax.set_aspect('equal')
    ax.legend()
    ax.grid()
plt.show()
 


# =============================================================================
# =============================================================================
# # 
# =============================================================================
# =============================================================================


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np

# Load dataset with xarray
r_path ='/home/shamu/HUB/eddy_trajectory_2.0exp_19930101_20191015.nc'
dataset = xr.open_dataset(r_path)


# Projection
proj=ccrs.PlateCarree()


#--------------------------------------------------------------------------------
# Follow a given eddy, plot its filtered track superimposed to its real track
#--------------------------------------------------------------------------------
# Select a specific eddy and save
subset = dataset.sel(obs=dataset.track==72553)
# Store selected data
subset.to_netcdf('eddy_trajectory_152014.nc')

# Position filtering with a rolling mean
windows_kwargs = dict(min_periods=7, obs=14, center=True)
new_lon = subset.longitude.rolling(** windows_kwargs).mean()
new_lat = subset.latitude.rolling(** windows_kwargs).mean()

# Create figure
fig = plt.figure()
# Create subplot
ax = fig.add_subplot(111, projection=proj)
# Plot the two tracks
ax.plot(subset.longitude, subset.latitude, color='r', label='original track', transform=proj)
ax.plot(new_lon, new_lat, color='b', label='filtered track', transform=proj)
# Active meridian/parallel
ax.gridlines()
# Active coastline
ax.coastlines()
# Legend
ax.legend()


#----------------------------------------------------------------------------------------------
# Select all eddies which go throught a given area and which have a lifespan more than 500 days
#----------------------------------------------------------------------------------------------

# Create figure
fig = plt.figure()
# Create subplot
ax = fig.add_subplot(111, projection=proj)

# Bounds of the area
lon_min, lon_max, lat_min, lat_max = 120, 140, -40, -34
# Draw area
ax.fill(
    [lon_min, lon_max, lon_max, lon_min, lon_min],
    [lat_min, lat_min, lat_max, lat_max, lat_min],
    color='coral',
    transform=proj,
    alpha=0.4,
    zorder=30)

# Select all observation in the area
subset = dataset.sel(
    obs=(dataset.longitude > lon_min) & (dataset.longitude < lon_max) &
    (dataset.latitude > lat_min) & (dataset.latitude < lat_max))
# Create a mask with all track which go throught the area
# Create the subset with the mask
subset = dataset.isel(obs=np.in1d(dataset.track, subset.track))
# Find all the track which are longer than 500 days
subset_lon_life =subset.sel(obs=subset.observation_number>500)
# Create the final subset
subset = subset.isel(obs=np.in1d(subset.track, subset_lon_life.track))
# Plot selected data
ax.scatter(
    subset.longitude,
    subset.latitude,
    c=subset.track,
    label='All tracks longer than 500 days',
    s=5,
    transform=proj,
    linewidth=0,
    cmap='Dark2')
# Active meridian/parallel
ax.gridlines()
# Active coastline 
ax.coastlines()
# Legend
ax.legend()

# Store subset to further analyse
subset.to_netcdf('eddy_trajectory_area_days_more500.nc')

# Display figure
plt.show()

# Selection of all event with an amplitude over 40 cm
subset = dataset.sel(obs=dataset.amplitude>40.)
# save in netcdf file with same properties as before
subset.to_netcdf('eddy_trajectory_amplitude_more40.nc')




