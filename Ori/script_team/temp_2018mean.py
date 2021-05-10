# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:12:29 2020

@author: psi36
"""

import os
import numpy as np
import time
import datetime as dt
import pandas as pd
from netCDF4 import Dataset
from scipy import io

LLD = 'E:/psi36/DATA/matfile/global/CDS_aviso/'

adt_17 = io.loadmat(LLD+'05adt_Monthlymean')

lon=io.loadmat(LLD+lon_type)['lon']
lat=io.loadmat(LLD+lat_type)['lat']

Mean_var = np.zeros([12,720,1440])

Mean_var[11,:,:] = B = np.nanmean(var[334:365,:,:],axis=0)


io.savemat('E:/psi36/DATA/matfile/global/CDS_aviso/05adt_Monthlymean.mat', {'Mean_var':Mean_var})


del Mean_var