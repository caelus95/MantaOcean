#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:32:13 2020

@author: shamu
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

r_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/Sigs/Yan_Sun_eemd.npy'
w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/Yan_Sun_Spatial_Mean/figs/imf5_01.png'

sig1 = np.load(r_path1)

S = sig1[:,5]

df = pd.DataFrame(S,columns=['value'])

t = pd.date_range('1993-01-01', periods = 324, freq = 1 * '1m').strftime('%Y-%m') # or freq = 3 * '1D'

df['date'] = t


# Get the Peaks and Troughs
data = df['value'].values

doublediff = np.diff(np.sign(np.diff(data)))
peak_locations = np.where(doublediff == -2)[0] + 1

doublediff2 = np.diff(np.sign(np.diff(-1*data)))
trough_locations = np.where(doublediff2 == -2)[0] + 1

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.labelfontsize"] = 2
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True
plt.rcParams['font.size'] = 18


# Draw Plot
plt.figure(figsize=(16,10), dpi= 80)
plt.plot('date', 'value', data=df, color='k',linewidth=3,label='imf5')
plt.axhspan(-np.std(df['value'])/2,np.std(df['value']/2), facecolor=[.7,.7,.7], alpha=0.5,label=r'imf5 < |$\frac{\sigma}{2}$|')
plt.scatter(df.date[peak_locations], df.value[peak_locations], marker=mpl.markers.CARETUPBASE, color='darkred', s=200, label='Peaks')
plt.scatter(df.date[trough_locations], df.value[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='darkblue', s=100, label='Troughs')

plt.text('1995-08',-.005,r'|$\frac{\sigma}{2}$| = 0.11')
# Annotate
for t, p in zip(trough_locations, peak_locations):
    plt.text(df.date[p], df.value[p]+.0025, df.date[p], horizontalalignment='center', color='darkred')
    plt.text(df.date[t], df.value[t]-.0038, df.date[t], horizontalalignment='center', color='darkblue')
# Decoration
plt.ylim(-.055,.055)
xtick_location = df.index.tolist()[::12*3]
xtick_labels = df.date.tolist()[::12*3]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=15, alpha=.7)
# plt.title('IMF05', fontsize=22)
plt.yticks(fontsize=15, alpha=.7)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)

plt.legend(loc='lower right')
plt.grid(axis='y', alpha=.3)
plt.savefig(w_path1,dpi=150)
plt.show()


