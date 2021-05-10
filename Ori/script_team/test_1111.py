# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:07:34 2020

@author: psi36
"""

import numpy as np
from scipy import io
import os


#adt = io.loadmat('F:/psi36/DATA/temp_var3/EastP_adt')

var1 = io.loadmat('F:/psi36/DATA/Kuroshio_NEC/Kuroshio_ori_ugos')
var2 = io.loadmat('F:/psi36/DATA/Kuroshio_NEC/Kuroshio_ori_vgos')

ugos = var1['ugos'][:]
vgos = var2['vgos'][:]

#adt1 = adt['adt'][:,:,:]

speed1 = (np.nanmean(ugos,axis=0)**2 + np.nanmean(vgos,axis=0)**2)**.5 
speed2 = np.nanmean(speed,axis=0)
speed = (ugos**2 + vgos**2)**.5

EKE = (ugos**2 + vgos**2)/2

mean_speed = np.nanmean(speed,axis=0)
mean_u = np.nanmean(ugos,axis=0)
mean_v = np.nanmean(vgos,axis=0)

mean_EKE = np.nanmean(EKE,axis=0)



speed_anomaly = speed - mean_speed
u_anomaly = ugos - mean_u
v_anomaly = vgos - mean_v

EKE = (ugos**2 + vgos**2)/2

speed_composite = composite1(speed_anomaly)
u_composite = composite1(u_anomaly)
v_composite = composite1(v_anomaly)

adt_composite = composite1(adt)

hyr_speed = speed_composite['hyr']
lyr_speed = speed_composite['lyr']

hyr_u = u_composite['hyr']
lyr_u = u_composite['lyr']

hyr_v = v_composite['hyr']
lyr_v = v_composite['lyr']

hyr_adt = adt_composite['hyr']
lyr_adt = adt_composite['lyr']


EKE_composite = composite1(EKE)

hyr_EKE = EKE_composite['hyr']
lyr_EKE = EKE_composite['lyr']


hyr_speed_aft = (hyr_u**2 + hyr_v**2)**.5
lyr_speed_aft = (lyr_u**2 + lyr_v**2)**.5


Ac,Bc = np.meshgrid(lon_rgnl,lat_rgnl)







