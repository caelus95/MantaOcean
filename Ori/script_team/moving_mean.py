# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 15:52:17 2019

@author: manta36
"""
# import modules
import os
import numpy as np
import time
import datetime as dt
import pandas as pd
from netCDF4 import Dataset
from scipy import io

# Difine variables
llcrnrlat = 16 # int  
urcrnrlat = 24 # int
llcrnrlon = 126 # int
urcrnrlon = 140 # int
dateST = 19930101 # int
var_name = 'vgos' # str
dateED = 20181231 # int
Directory = 'F:/psi36/DATA/ncfile/CDS/nc/' # str
Mean_interval = 60 #int


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

    DataNs=dt.date.toordinal(dt.date(DYs,DMs,DDs))
    DataNe=dt.date.toordinal(dt.date(DYe,DMe,DDe))
    DNs=dt.date.toordinal(dt.date(Ys,Ms,Ds))
    DNe=dt.date.toordinal(dt.date(Ye,Me,De))
    
    I_DataNs_DNs = DNs - DataNs  
    I_DataNs_DNe = DataNe - DNe 
    
    return [I_DataNs_DNs,I_DataNs_DNe]

def percentage1(i,L):
    if i == 0 : 
        print('Processing......0%')
    elif i%L == round(L*0.10) :
        print('Processing......10%')
    elif i%L == round(L*0.25) :    
        print('Processing......25%')
    elif i%L == round(L*0.33) :
        print('Processing......33%')
    elif i%L == round(L*0.50) :
        print('Processing......50%')
    elif i%L == round(L*0.66) :
        print('Processing......66%')
    elif i%L == round(L*0.75) :
        print('Processing......75%')
    elif i%L == round(L*0.95) :
        print('Processing......95%')
    elif i==L-1 : 
        print('Processing......100%')

t1, t2 = date_interval(DataST,DataED,dateST,dateED)

nc_list[:t1] = [] 

if t2 == 0 :
    pass
else:
    nc_list[-t2:] = []
    
dataset = Dataset(nc_list[0])

ELat = dataset.variables['latitude'][:]
ELon = dataset.variables['longitude'][:]

lat_co = np.where((ELat >= llcrnrlat) & (ELat <= urcrnrlat))
lon_co = np.where((ELon >= llcrnrlon) & (ELon <= urcrnrlon))

lat_rgnl = ELat[lat_co]
lon_rgnl = ELon[lon_co]

var = np.zeros((len(nc_list), len(lat_co[0]), len(lon_co[0])))
print('.')
print('..')
print('>>Reading NCs...')
print('')
t = time.time()
i = 0
while i <= len(nc_list) - 1:
    dataset = Dataset(nc_list[i])
    var[i, :, :] =  A = dataset.variables[var_name][0, lat_co[0], lon_co[0]]#.data[0]
    percentage1(i,len(nc_list))
    i = i + 1
         
var[var<-10] = np.nan

elapsed = time.time() - t      
print('Elapsed Time ==> ',round(elapsed,2),'(s)')  
print('variable : var ==> ', var_name)
print('var : var(time,latitude,longitude) ==> ',[len(nc_list),len(lat_co[0]),len(lon_co[0])]) 
print('Period : ',dateST,'~',dateED) 
print('>>Complete')    

# 
me,at,on = np.shape(var)

var_MovingMean = np.zeros([me,at,on])
t = time.time()
print('.')
print('..')
print('>>Calculating MovingMean...')
print('')
for i in range(at):
    percentage1(i,len(np.arange(at)))
    for j in range(on):
        dataframe_im = pd.DataFrame(np.squeeze(var[:,i,j]))
        temp_var = dataframe_im.rolling(window=Mean_interval,center=True).mean()
        var_MovingMean[:,i,j] = np.squeeze(temp_var)
        
print('Elapsed Time ==> ',round(elapsed,2),'(s)')  
print('variable : var_MovingMean ==> ', 'MovingMean_daily')
print('var : var(time,latitude,longitude) ==> ',[len(nc_list),len(lat_co[0]),len(lon_co[0])]) 
print('Period : ',dateST,'~',dateED) 
print('>>Complete')    

'''
nan처리된곳 데이터 / 날짜에서 자르기
아 존나 귀찮네
'''

tS = dt.date.toordinal(dt.date(int(str(dateST)[0:4]),int(str(dateST)[4:6]),int(str(dateST)[6:8])))
tE = dt.date.toordinal(dt.date(int(str(dateED)[0:4]),int(str(dateED)[4:6]),int(str(dateED)[6:8])))
d1 = pd.DataFrame(np.arange(tS,tE+1))

'''
날짜 _ 인덱스 변수를 만들어서 판다로 처리하여
좌표를 

'''

d2 = np.zeros([len(d1),2])

n = 0
for i in d1[0]:
    d3 = dt.datetime.fromordinal(int(i))
    d2[n,0] = d3.strftime("%Y%m%d")
    n+=1
    
d2[:,1] = np.arange(len(d1))    


'''
co1 = pd.DataFrame(index = np.arange(len(d1)))
co1.insert(0, 'coordinates', list , allow_duplicates = False)

n = -1 ; m = 0 
for i in d2[:,1]:
    #print(int(i))
    n+=1
    if int(i) == len(d1)-1:
       co1.xs(int(i))['coordinates'] = d2[int(i-(n)):int(i+1),1]
    elif str(d2[int(i),0])[4:6] != str(d2[int(i+1),0])[4:6]:
        co1.xs(int(i))['coordinates'] = d2[int(i-(n)):int(i+1),1] 
        print(str(d2[int(i),0])[0:] )
        print(n+1)
        print(int(i-(n-1)))
        #print(str(d2[int(i),0])[4:6])
        m+=1 ; n = -1
    
        
    
      print(co1.xs(0)['column']) = 10
    
'''
# coordinates of each months 
co1 = {}

n = -1 ; m = 0 
for i in d2[:,1]:
    #print(int(i))
    n+=1
    if int(i) == len(d1)-1:
       co1[int(m)] = d2[int(i-(n)):int(i+1),1]
    elif str(d2[int(i),0])[4:6] != str(d2[int(i+1),0])[4:6]:
        co1[int(m)] = d2[int(i-(n)):int(i+1),1] 
        print(str(d2[int(i),0])[0:] )
        print(n+1)
        print(int(i-(n-1)))
        #print(str(d2[int(i),0])[4:6])
        m+=1 ; n = -1

# nanmean data

MovingMean_Monthly_data = np.zeros([len(co1),at,on])
m=0
for i in co1:
    MovingMean_Monthly_data[m] = np.nanmean(var_MovingMean[int(co1[i][0]):int(co1[i][-1]),:,:],axis=0,out=None)
    m+=1 
    
'''    
io.savemat('E:/psi36/DATA/temp_var/ET_eddy_ugos.mat', {'ugos':MovingMean_Monthly_data})
'''



#var_c = var[:] - 273.15 # substract offset (kelvin ==> celsius)
