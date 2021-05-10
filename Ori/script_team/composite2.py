# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:34:34 2019

@author: psi36
"""

import numpy as np
from scipy import io

var = var_dic['anomaly']

hyr = [[1996, 1997, 1998, 2003, 2004, 2008, 2009, 2016],[3,4,5,10,11,15,16,23]]
lyr= [[1993, 2000, 2001, 2005, 2006, 2012, 2013, 2014],[0,7,8,12,13,19,20,21]]

var_hyr = var[hyr[1],:,:]
var_lyr = var[lyr[1],:,:]


'''
DJF_hyr = DJF[hyr[1],:,:]
DJF_lyr = DJF[lyr[1],:,:]
'''