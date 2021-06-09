#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 01:05:44 2021

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

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
Sig_set,Corr_map,Annual_mean = sig_pro(r_path,['1993-01-01',324,300],Standard=True)

Sig_set['dates'] = pd.to_datetime(Sig_set.index).strftime('%Y-%m')


tmp_MEIP = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm >= 0)
tmp_MEIN = Sig_set.MEIv2_2Y_Rm.where(Sig_set.MEIv2_2Y_Rm < 0)

# ------------  fig ----------------
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['axes.linewidth'] = 3
# plt.rcParams['axes.grid'] = False
plt.rcParams['xtick.labeltop'] = False
plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['ytick.labelleft'] = True



Sig_set.keys()

plt.figure(1,figsize=(16,7),dpi=80)
ax = plt.gca()

plt.plot(Sig_set.dates,Sig_set.ADT_index_2Y_Rm, label='Kuroshio_Rm (2Y Runing mean)',color='k',linewidth=2.5)
plt.plot(Sig_set.dates,Sig_set.PDO_2Y_Rm, label='EKE_wang_pc1_2Y_Rm (2Y Runing mean)',color='darkred',linewidth=2.5)
# plt.plot(Sig_set.dates,Sig_set.WP_2Y_Rm, label='WP (2Y Runing mean)',color='g',linewidth=2.5)
# plt.plot(Sig_set.dates,Sig_set.PTO_index_2Y_Rm, label='PTO (2Y Runing mean)',color='r',linewidth=2.5)
plt.plot(Sig_set.dates,Sig_set.PNA_2Y_Rm, label='PNA (2Y Runing mean)',color='b',linewidth=2.5)

plt.fill_between(Sig_set.dates, tmp_MEIP, color="lightpink",
                      alpha=0.5, label='El-nino',zorder=0)
plt.fill_between(Sig_set.dates, tmp_MEIN, color="skyblue",
                      alpha=0.5, label='La nina',zorder=1)
    
plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)

# plt.bar(Annual_mean.index,Annual_mean.MEIv2, label='MEIv2',color=[.9,.9,.9],linewidth=2.5,zorder=0,alpha=.7)

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3)
plt.axvline(x=Sig_set.dates[20],color='k',linewidth=3,linestyle='--',alpha=.9)




plt.figure(1,figsize=(16,6),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.title('a) Indexes (2Y running mean)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
plt.plot(Sig_set.dates,Sig_set.ADT_index_2Y_Rm, label='KVTe index (Yan & Sun 2015)',color='k',linewidth=3,zorder=10)
plt.plot(Sig_set.dates,Sig_set.EKE_qiu_10_30_120_250_2Y_Rm, label='EKE (Qiu 2013)',color='darkred',linewidth=3,zorder=9)
plt.plot(Sig_set.dates,Sig_set.PDO_2Y_Rm, label='PDO ',color='darkblue',linewidth=2.5,zorder=8)
plt.plot(Sig_set.dates,Sig_set.PTO_index_2Y_Rm, label='PTO (Chang & Oey 2011)',color='C1',linewidth=2.5)
plt.plot(Sig_set.dates,Sig_set.WP_2Y_Rm, label='WP ',color='green',linewidth=2.5,zorder=7)

plt.fill_between(Sig_set.dates, tmp_MEIP, color="lightpink",
                  alpha=0.5, label='El-nino',zorder=0)
plt.fill_between(Sig_set.dates, tmp_MEIN, color="skyblue",
                  alpha=0.5, label='La nina',zorder=1)

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=15)
plt.axvline(x=Sig_set.dates[146],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=15)
plt.axvline(x=Sig_set.dates[242],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=15)


# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.dates.tolist()[::12*2]
xtick_labels = Sig_set.dates.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower right',fontsize=10)
plt.grid(axis='y', alpha=.6)
plt.tight_layout()
# plt.savefig('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/Kuroshio_index')
plt.show()



# =============================================================================
# 
# =============================================================================

ADT_index = Sig_set.ADT_index
ADT_index_2Y_Rm = Sig_set.ADT_index_2Y_Rm
LPF
filtered_sig
# =============================================================================
# LPF
# =============================================================================

import numpy as np
from scipy.signal import butter, filtfilt 
import matplotlib.pyplot as plt 

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff/nyq
    b,a = butter(order, normal_cutoff, btype='low', analog=False)
    y=filtfilt(b, a, data)
    return y 

Signal = ADT_index.values.reshape(-1)
L = len(Signal) 
T = 6 #period(time interval)
fs = L/T #rate(total number of samples/period)
cutoff = 1 #cutoff frequency

nyq= 0.5*fs
order=2 #filter order
n= int(T*fs) #total number of samples 

LPF = butter_lowpass_filter(Signal, cutoff, fs, order) 

plt.plot(LPF)


# =============================================================================
# 
# =============================================================================


import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

r_path = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/signal/YS_dif.npy'
w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/figs/'

sig = ADT_index.values.reshape(-1)
# sig = A# Ori_Rm_seasonality # np.array(SIG) # np.array(seasonality) # sig[:,0]

# sig=imf5
time_vec = np.arange(324)

#신호를 생성합니다.
# Seed the random number generator
np.random.seed(1234)

period = 1.


plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')

#FFT의 Power를 계산합니다.
# The FFT of the signal
sig_fft = fftpack.fft(sig)

# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=1)

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

# Find the peak frequency: we can focus on only the positive frequencies
pos_mask = np.where(sample_freq > 0)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]

# Check that it does indeed correspond to the frequency that we generate
# the signal with 
np.allclose(peak_freq, 1./period)

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5])
plt.title('Peak frequency')
plt.plot(freqs[:8], power[:8])
plt.setp(axes, yticks=[])
# scipy.signal.find_peaks_cwt can also be used for more advanced  peak detection



#모든 high frequencies를 제거합니다.
high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

plt.figure(figsize=(6, 5))
plt.title('Removed high freq (fft)',fontsize=28,fontweight='bold')
plt.plot(time_vec, sig, label='Original signal',color=[.7,.7,.7])
plt.plot(time_vec, filtered_sig, linewidth=3, label='Filtered signal',color='k')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.legend(loc='best')
# plt.savefig(w_path1+'fft_removed_high_freq')
plt.show()

# =============================================================================
# figure
# =============================================================================

ADT_index = Sig_set.ADT_index
ADT_index_2Y_Rm = Sig_set.ADT_index_2Y_Rm
LPF
filtered_sig
CEEMD = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/ceemd/ADT_index_ceemd.npy')
# CEEMD = np.load('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/ceemd/first_5_ceemdan.npy')

ADT_ceemd = CEEMD[:,4]

plt.figure(1,figsize=(14,8),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.title('a) KVTe index (Yan & Sun 2015)', fontproperties='',loc='left',pad=15,  fontsize=28,fontweight='regular')
plt.plot(Sig_set.dates,ADT_index, label='KVTe index (Yan & Sun 2015)',color=[.7,.7,.7],linewidth=2,zorder=1)
plt.plot(Sig_set.dates,ADT_index_2Y_Rm, label='Running mean',color='k',linewidth=3.5,zorder=2)
plt.plot(Sig_set.dates,LPF, label='LPF ',color='C8',linewidth=3.5,zorder=3)
# plt.plot(Sig_set.dates,Sig_set.PTO_index_2Y_Rm, label='PTO (2Y Runing mean)',color='C1',linewidth=2.5)
plt.plot(Sig_set.dates,filtered_sig, label='FFT',color='darkred',linewidth=3.5,zorder=4)
plt.plot(Sig_set.dates,ADT_ceemd*10, label='CEEMDAN',color='C0',linewidth=3.5,zorder=5)

plt.ylim(-2.5,2.5)

plt.axhline(y=0,color='k',linewidth=3,linestyle='--',alpha=.3,zorder=6)
# plt.axvline(x=Sig_set.dates[146],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=7)
# plt.axvline(x=Sig_set.dates[242],color='k',linewidth=3,linestyle='--',alpha=.9,zorder=7)


# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.dates.tolist()[::12*2]
xtick_labels = Sig_set.dates.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=20, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower right',fontsize=14)
plt.grid(axis='y', alpha=.6)
plt.tight_layout()
# plt.savefig('/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/YS_Kuroshio_index')
plt.show()








