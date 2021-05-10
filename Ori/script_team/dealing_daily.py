# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:38:56 2019

@author: navys
"""
# import modules
import os
import numpy as np
import time
from datetime import date
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# Difine variables
mLat = -15 # int  
MLat = 45 # int
mLon = 80 # int
MLon = 180 # int
dateST = 19930105 # int
dateED = 19940719 # int
var_name = 'analysed_sst' # str
Directory = 'E:/DATA/CDS/nc/' # str

os.chdir(Directory)
dir_list = os.listdir(Directory)
nc_list = [file for file in dir_list if file.endswith(".nc")]

DataST = str(''.join(list(filter(str.isdigit, nc_list[0]))))[0:8]
DataED = str(''.join(list(filter(str.isdigit, nc_list[-1]))))[0:8]

def date_interval(DataST,DataED,dateST,dateED):
    DYs = int(str(DataST)[0:4])
    DMs = int(str(DataST)[4:6])
    DDs = int(str(DataST)[6:8])
    
    DYe = int(str(DataED)[0:4])
    DMe = int(str(DataED)[4:6])
    DDe = int(str(DataED)[6:8])
    
    Ys = int(str(dateST)[0:4])
    Ms = int(str(dateST)[4:6])
    Ds = int(str(dateST)[6:8])

    Ye = int(str(dateED)[0:4])
    Me = int(str(dateED)[4:6])
    De = int(str(dateED)[6:8])

    DataNs=date.toordinal(date(DYs,DMs,DDs))
    DataNe=date.toordinal(date(DYe,DMe,DDe))
    DNs=date.toordinal(date(Ys,Ms,Ds))
    DNe=date.toordinal(date(Ye,Me,De))
    
    I_DataNs_DNs = DNs - DataNs  
    I_DataNs_DNe = DataNe - DNe 
    
    return [I_DataNs_DNs,I_DataNs_DNe]

t1, t2 = date_interval(DataST,DataED,dateST,dateED)

nc_list[:t1] = [] 

if t2 == 0 :
    print(t2)
else:
    nc_list[-t2:] = []
        
dataset = Dataset(nc_list[0])

ELat = dataset.variables['latitude'][:]
ELon = dataset.variables['longitude'][:]

Lat_CO = np.where((ELat >= mLat) & (ELat <= MLat))
Lon_CO = np.where((ELon >= mLon) & (ELon <= MLon))

Lat_RE = ELat[Lat_CO]
Lon_RE = ELon[Lon_CO]

var = np.zeros((len(nc_list), len(Lat_CO[0]), len(Lon_CO[0])))
print('.')
print('..')
t = time.time()
i = 0
while i <= len(nc_list) - 1:
    dataset = Dataset(nc_list[i])
    var[i, :, :] =  A = dataset.variables[var_name][0, Lat_CO[0], Lon_CO[0]]#.data[0]
    i = i + 1

    if i%len(nc_list) == 1 : 
        print('Processing......0%')
    elif i%len(nc_list) == round(len(nc_list)*0.10) :
        print('Processing......10%')
    elif i%len(nc_list) == round(len(nc_list)*0.25) :    
        print('Processing......25%')
    elif i%len(nc_list) == round(len(nc_list)*0.33) :
        print('Processing......33%')
    elif i%len(nc_list) == round(len(nc_list)*0.50) :
        print('Processing......50%')
    elif i%len(nc_list) == round(len(nc_list)*0.66) :
        print('Processing......66%')
    elif i%len(nc_list) == round(len(nc_list)*0.75) :
        print('Processing......75%')
    elif i%len(nc_list) == round(len(nc_list)*0.95) :
        print('Processing......95%')
        
elapsed = time.time() - t      
print('Processing......100%')    
print('Elapsed Time ==> ',round(elapsed,2),'(s)')  
print('variable : var ==> ', var_name)
print('var : var(time,latitude,longitude) ==> ',[len(nc_list),len(Lat_CO[0]),len(Lon_CO[0])]) 
print('Period : ',dateST,'~',dateED) 
print('END')    

var_c = var[:] - 273.15 # substract offset (kelvin ==> celsius)

## basemap

m = Basemap(projection='cyl',llcrnrlat=mLat,urcrnrlat=MLat,\
            llcrnrlon=mLon,urcrnrlon=MLon,resolution='c')
lon2, lat2 = np.meshgrid(Lon_RE,Lat_RE)
x, y = m(lon2, lat2)
fig = plt.figure(figsize=(15,7))
m.fillcontinents(color='gray',lake_color='k')
m.drawcoastlines()
m.drawparallels(np.arange(-80.,81.,20.),labels=[True,True,True,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,20.),labels=[False,False,False,True],dashes=[2,2])
m.drawmapboundary(fill_color='white')
cs = m.contourf(x,y,np.nanmean(var_c,axis=0),20,cmap=plt.cm.get_cmap('jet'))
h = plt.colorbar(label='℃');
plt.title(' Total mean of SST ', fontname='Comic Sans MS', fontsize=24)





