#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:22:28 2021

@author: caelus
"""

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/'
NPGO = xr.open_dataset(r_path+'inpgo.nc',decode_times=False)


NPGO_index = NPGO.NPGO.values

dates = pd.date_range('1950-01-15', periods = 828,freq = 1 * '1m')


plt.plot(dates[-312:],NPGO_index[-312:])

tmp_npgo = NPGO_index[-312:]

tmp = np.zeros(12)
tmp[tmp==0] = np.nan

npgo_index = np.concatenate([tmp_npgo,tmp])

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/NPGO',npgo_index)


ADT_index = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/ADT_index.npy')


#  𝟎.𝟑𝟏 ×𝑺𝑳𝑫+𝟔.𝟓𝟓
KVTe = 0.31*ADT_index*100+6.55

plt.plot(dates[-324:],KVTe)

np.mean(KVTe)
np.std(KVTe)

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/KVTe_index',KVTe)

# NP
NP = pd.read_csv('/home/caelus/wormhole/CP/NP_NPGO.csv',header=0)

NP_index = NP['NP'].values

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/NP',NP_index)


# ALBSA
ALBSA = pd.read_csv('/home/caelus/wormhole/CP/albsa.csv',header=0)

ALBSA = ALBSA[-348:]
ALBSA = ALBSA[:-24]


ALBSA_index = ALBSA.iloc[:,1].values

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/ALBSA',ALBSA_index)




# Nino4
Nino4 = pd.read_csv('/home/caelus/wormhole/CP/NINO4.csv',header=None)

Nino4 = Nino4.values.reshape(-1)



ALBSA_index = ALBSA.iloc[:,1].values

np.save('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/NINO4',Nino4)












