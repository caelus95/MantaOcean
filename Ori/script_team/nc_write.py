#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#=======================================================================
# name: nc.py
#
# category: python script
#
# description:
#   This is a sample program to read and write a netCDF file with
#   Python.
#
# reference:
#   http://unidata.github.io/netcdf4-python/
#
# author: M. Yoshimori (myoshimo AT ees.hokudai.ac.jp)
#=======================================================================

from netCDF4 import Dataset
from numpy import dtype

#-----------------
# read netCDF file
#-----------------

# open a netCDF file to read
filename = "av_an_sfc_y1979-y2017_mon.nc"
ncin = Dataset(filename, 'r', format='NETCDF4')

# check netCDF file format
#print(ncin.file_format)

# get axis data
#print(ncin.dimensions.keys())
#print(ncin.dimensions['time'])
tin = ncin.variables['time']
latitude = ncin.variables['latitude']
longitude = ncin.variables['longitude']

# get length of axis data
#ntime = len(tin)
nlat = len(latitude)
nlon = len(longitude)

# print axis
#print(tin[:])
#print(latitude[:])
#print(longitude[:])

# get variables
#print(ncin.variables.keys())
#print(ncin.variables['t2m'])

# read data
vin = ncin.variables['t2m']
#print(vin.long_name)
#print(vin.units)

#------------------
# write netCDF file
#------------------

# open a netCDF file to write
ncout = Dataset('testout.nc', 'w', format='NETCDF4')

# define axis size
ncout.createDimension('time', None)  # unlimited
ncout.createDimension('lat', nlat)
ncout.createDimension('lon', nlon)

# create time axis
time = ncout.createVariable('time', dtype('double').char, ('time',))
time.long_name = 'time'
time.units = 'hours since 1990-01-01 00:00:00'
time.calendar = 'standard'
time.axis = 'T'

# create latitude axis
lat = ncout.createVariable('lat', dtype('double').char, ('lat'))
lat.standard_name = 'latitude'
lat.long_name = 'latitude'
lat.units = 'degrees_north'
lat.axis = 'Y'

# create longitude axis
lon = ncout.createVariable('lon', dtype('double').char, ('lon'))
lon.standard_name = 'longitude'
lon.long_name = 'longitude'
lon.units = 'degrees_east'
lon.axis = 'X'

# create variable array
vout = ncout.createVariable('t2m', dtype('double').char, ('time', 'lat', 'lon'))
vout.long_name = '2 metre temperature'
vout.units = 'K'

# copy axis from original dataset
time[:] = tin[:]
lon[:] = longitude[:]
lat[:] = latitude[:]
vout[:] = vin[:]

# close files
ncin.close()
ncout.close()
