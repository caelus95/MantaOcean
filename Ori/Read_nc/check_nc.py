#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:07:16 2021

@author: caelus
"""

PKG_path = '/home/caelus/dock_1/Working_hub/LGnDC_dep/python_cent/MantaPKG/'
import sys 
sys.path.append(PKG_path)
from Manta_Signals.procc_index import sig_pro, linearRegress4Cube
from Manta_Signals.utility import nc2npy
import os
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil


r_path1 = '/home/caelus/dock_2/psi36/DATA/ncfile/GRSST/nc_files/'

nc_list = np.sort([file for file in os.listdir(r_path1) if file.endswith(".nc")])


error_list = []
for i in tqdm(nc_list):
    try:
        xr.open_dataset(r_path1+i)
    except :
        error_list.append(i)
        shutil.move(r_path1+i,r_path1+'Error_files/')
        













