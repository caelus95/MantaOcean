#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 20:49:47 2021

@author: caelus
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 18:33:12 2021

@author: caelus
"""



%matplotlib auto

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, ones, pi
from scipy import cos, sin
from scipy.fftpack import fft, fftfreq, ifft




sig2 = Sig_set.ET_Speed_pc1.values
sig1 = Sig_set.ADT_index.values


sig1 = sig1.reshape(-1)
# sig = sig[:240]
# sig = RM.values.reshape(-1)
x= np.arange(len(sig1))
n = len(sig1)
freqs1 = fftfreq(n)    # 필요한 모든 진동수를 만든다.
mask1 = freqs1 > 0    # 절반의 값을 무시
nwaves = freqs1*n    # 도메인 길이에 따른 파수
fft_vals = fft(sig1)    # FFT 계산
fft_norm1 = fft_vals*(1.0/n)    # FFT 계산된 결과를 정규화
fft_theo1 = 2.0*abs(fft_norm1)    # 푸리에 계수 계산

sig2 = sig2.reshape(-1)
# sig = sig[:240]
# sig = RM.values.reshape(-1)
x= np.arange(len(sig2))
n = len(sig2)
freqs2 = fftfreq(n)    # 필요한 모든 진동수를 만든다.
mask2 = freqs2 > 0    # 절반의 값을 무시
nwaves = freqs2*n    # 도메인 길이에 따른 파수
fft_vals = fft(sig2)    # FFT 계산
fft_norm2 = fft_vals*(1.0/n)    # FFT 계산된 결과를 정규화
fft_theo2 = 2.0*abs(fft_norm2)    # 푸리에 계수 계산




date = pd.date_range('1993-01-01', periods = 324,freq = 1 * '1m').strftime('%Y-%m')
# 1.   
plt.figure(figsize=(16,7))
plt.subplot(211)
plt.bar(freqs1[mask1]*n, fft_theo1[mask1], label="true fft values",color='r')
plt.title("True FFT values")
# plt.xlim([0,1])
# plt.axhspan(-0, .02, facecolor='0.5', alpha=0.2)
plt.axhline(y=0, linewidth=1, color='r')
plt.legend()

# 2. 푸리에 계수
plt.figure(figsize=(16,10))
plt.bar(freqs1[mask1]*n, fft_theo1[mask1], label="ADT_index",color='darkred',alpha=0.3)
plt.bar(freqs2[mask2]*n, fft_theo2[mask2], label="first_5",color='darkblue',alpha=.3)
# plt.title("True FFT values")
plt.xlim([0,80])
plt.axhspan(-0, .02, facecolor='0.5', alpha=0.2)
plt.axhline(y=0, linewidth=1, color='r')
plt.yticks(fontsize=18, alpha=.7)
plt.xticks(fontsize=18, alpha=.7)
plt.legend(fontsize=24)
plt.show()



plt.figure(figsize=(16,10))
plt.bar(324/12/(freqs1[mask1]*n), fft_theo1[mask1], label="ADT_index",color='darkred',alpha=0.3)
plt.bar(324/12/(freqs2[mask2]*n), fft_theo2[mask2], label="first_5",color='darkblue',alpha=.3)
# plt.title("True FFT values")
# plt.xlim([0,1])
plt.axhspan(-0, .02, facecolor='0.5', alpha=0.2)
plt.axhline(y=0, linewidth=1, color='r')
plt.yticks(fontsize=18, alpha=.7)
plt.xticks(fontsize=18, alpha=.7)
plt.legend(fontsize=24)
plt.show()






plt.stem(324/12/(freqs[mask]*n), fft_theo[mask], label="true fft values")
# plt.xlim([0,25])






plt.figure(figsize=(16,7))
plt.subplot(211)
plt.plot(date, Sig_set.ADT_index, color='k', label='ADT_index',zorder=2)
plt.plot(date, Sig_set.ET_Speed_pc1, color=[.7,.7,.7], label='ET_Speed_pc1',zorder=1)
plt.plot(date, Sig_set.ADT_index_2Y_Rm, color='darkred', label='ADT_index_2Y_Rm',linewidth=2.5,zorder=3)
plt.plot(date, Sig_set.ET_Speed_pc1_2Y_Rm, color='darkblue', label='ET_Speed_pc1_2Y_Rm',linewidth=2.5,zorder=4)

# plt.plot(x, origin, color='g', label='IFFT')
# plt.title("Original & IFFT Signal")
plt.legend(fontsize=18)
xtick_location = date.tolist()[::12*2]
xtick_labels = date.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=0, fontsize=12, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=.3)

# 2. 푸리에 계수
plt.subplot(212)
plt.bar(freqs1[mask1]*n, fft_theo1[mask1], label="FFT ADT index",color='darkred',alpha=0.3)
plt.bar(freqs2[mask2]*n, fft_theo2[mask2], label="FFT ET_Speed_pc1",color='darkblue',alpha=.3)
# plt.title("True FFT values")
plt.xlim([0,80])
plt.axhspan(-0, .02, facecolor='0.5', alpha=0.2)
plt.axhline(y=0, linewidth=1, color='r')
plt.yticks(fontsize=12, alpha=.7)
plt.xticks(fontsize=12, alpha=.7)
plt.grid(True, axis='y',alpha=.3)
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend()
plt.show()











