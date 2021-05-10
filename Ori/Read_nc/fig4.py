#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:28:38 2020

@author: shamu
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

r_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/Sigs/Yan_Sun_eemd.npy'
r_path2 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/Sigs/Yan_Sun_eemd_1.npy'
r_path3 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/Sigs/Yan_Sun_eemd_2.npy'
r_path4 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/Sigs/Yan_Sun_eemd_3.npy'

w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/figs/'

sig1 = np.load(r_path1)
sig2 = np.load(r_path2)
sig3 = np.load(r_path3)
sig4 = np.load(r_path4)


date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True
plt.rcParams['font.size'] = 18


plt.figure(figsize=(16,10), dpi= 80)
plt.plot(date,sig1[:,5],color='r',linewidth=2.5,label=1)
plt.plot(date,sig2[:,5],color='b',linewidth=2.5,label=2)
plt.plot(date,sig3[:,5],color='g',linewidth=2.5,label=3)
plt.plot(date,sig4[:,5],color='k',linewidth=2.5,label=4,linestyle='-.')

plt.ylim(-.05,.05)
xtick_location = date.tolist()[::12]
xtick_labels = date.tolist()[::12]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=45, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper left')
plt.grid(axis='y', alpha=.3)
# plt.savefig(w_path1+'decompo')
plt.show()