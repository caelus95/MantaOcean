#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 20:25:19 2021

@author: caelus
"""
import numpy as np
# import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

r_path1 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/latest/sigs/first_5.npy'
r_path2 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Validations/eemd_results/eemd_1000.npy'
r_path3 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Validations/eemd_results/Huang_eemd_1000.npy'
r_path4 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Validations/eemd_results/ceemd_0_1000.npy'
r_path5 = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Validations/eemd_results/ceemd_1_1000.npy'

w_path =  '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/Validations/figs/eemd_all_compare/'

sig_1 = np.load(r_path5)


plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True
plt.rcParams['font.size'] = 18
plt.rcParams['lines.linewidth'] = 2


date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')



fig, ax = plt.subplots(7, 1,figsize=(14,8))
ax[0].set_title(' CEEMDAN 1', fontsize=26,fontweight='bold')

ax[0].plot(sig_1[:,0])
ax[1].plot(sig_1[:,1])
ax[2].plot(sig_1[:,2])
ax[3].plot(sig_1[:,3])
ax[4].plot(sig_1[:,4])
ax[5].plot(sig_1[:,5])
ax[6].plot(date,sig_1[:,6])
# ax[7].plot(date,sig_1[:,7])

for i in range(6):
    ax[i].axes.xaxis.set_visible(False)
    ax[i].axes.yaxis.set_visible(False)
ax[i+1].axes.yaxis.set_visible(False)


xtick_location = date.tolist()[::12*3]
xtick_labels = date.tolist()[::12*3]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=15, alpha=.7)
plt.yticks(fontsize=15, alpha=.7)

# Lighten borders
# plt.gca().spines["top"].set_alpha(.0)
# plt.gca().spines["bottom"].set_alpha(.3)
# plt.gca().spines["right"].set_alpha(.0)
# plt.gca().spines["left"].set_alpha(.3)

# plt.legend(loc='lower right')
plt.grid(axis='y', alpha=.3)
plt.savefig(w_path+'ceemd_1.jpg',dpi=150)
plt.show()

# ax[1].set_title("Sine function")
# ax[2].set_title("Cosine function")
# ax[3].set_title("Sigmoid function")
# ax[4].set_title("Exponential function")






