#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 22:10:36 2020

@author: shamu
"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

r_path = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/signal/YS_dif.npy'
w_path1 = '/home/shamu/mangrove1/Working_hub/DATA_dep/Kuroshio/figs/'

sig = SIG.values.reshape(-1)
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





