# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:54:49 2019

@author: manta36
@title : Extract_data 
@Version : mat01
"""
import numpy as np
from scipy import io
import os
from numpy import hstack

Directory = 'E:/psi36/DATA/matfile/global/CDS_aviso/'
LLD ='F:/psi36/DATA/matfile/global/'
var_name = 'adt'
var_type = 'Monthlymean' 
lon_type = 'lon' 
lat_type = 'lat' 

llcrnrlat=-10
urcrnrlat=40
llcrnrlon=110
urcrnrlon=260
t_len = 312


lon=io.loadmat(LLD+lon_type)['lon']
lat=io.loadmat(LLD+lat_type)['lat']

lat_co=hstack(np.where((lat>=llcrnrlat)&(lat<=urcrnrlat))[0])
lon_co=hstack(np.where((lon>=llcrnrlon)&(lon<=urcrnrlon))[0])

lat_rgnl=lat[lat_co]
lon_rgnl=lon[lon_co]

var1=np.zeros((t_len, len(lat_co), len(lon_co)))

dir_list=os.listdir(Directory)

mat_list=[file for file in sorted(dir_list) if file.endswith(var_name+'_'+var_type+'.mat')]

n = 0
for i in mat_list :
    Mean_var = io.loadmat(Directory+i)['Mean_var']
    L = len(Mean_var)
    var1[n:n+L] = Mean_var[:,lat_co[0]:lat_co[-1]+1,lon_co[0]:lon_co[-1]+1]
    #var1[n:n+L] = Mean_var[:,Lat_CO[0],Lon_CO[0]]
    n = n+L
    
    mean=np.nanmean(var1,axis=0,out=None)

var=var1-mean

var_dic={'data':var1,'mean':mean,'anomaly':var}


'''
# save as mat(matlab) format
# double ==> 
io.savemat('F:/DATA/matfile/test/np_vector2.mat', {'var_dic':var_dic})
#struct ==> 
io.savemat('F:/DATA/matfile/np_vector.mat', var_dic)
'''




