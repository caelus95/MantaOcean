#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 15:38:42 2021

@author: caelus
"""

import numpy as np
import matplotlib.pyplot as plt

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'

sig = np.load(r_path+'first_5.npy')

time = np.linspace(10,1,1000)

s1 = 2*np.sin(20*np.pi*time)
s2 = np.sin(40*np.pi*time)
s3 = 0.5*np.sin(60*np.pi*time)
s4 = 1.5*np.sin(80*np.pi*time)

s = s1+s2+s3+s4


strength = np.fft.fft(sig.reshape(-1),norm='ortho')
strength = abs(strength)
frequency = np.fft.fftfreq(len(sig),1)

plt.plot(frequency, strength)
plt.show()







plt.xlim(0,0.04,10)
plt.ylim(0,10,200)
plt.grid()
plt.plot(frequency,strength)
plt.show()



S = pd.DataFrame({'sig':sig})


import pandas as pd

WY = 2
RM = S.rolling(window=12*WY,center=True).mean()

RM.dropna(inplace=True)
plt.plot(RM.values)

s = RM.values



# =============================================================================
# 
# =============================================================================

import matplotlib.pyplot as plt
from numpy import arange, ones, pi
from scipy import cos, sin
from scipy.fftpack import fft, fftfreq, ifft


n   = 100          # 한 파장당 지점 개수
Lx  = 10           # 한 파장의 거리 (혹은 시간)
L   = Lx/(2*pi)    # 파장
omg = 2.0*pi/Lx    # 각진동수

x   = arange(0, n)/Lx       # x축 : n개의 지점, Lx의 길이를 한 파장으로 가정
y1  = 5.0*sin( 2.0*omg*x)   # 파수  2에 해당하는 사인파
y2  = 4.0*cos( 3.0*omg*x)   # 파수  3에 해당하는 코사인파
y3  = 1.0*sin(10.0*omg*x)   # 파수 10에 해당하는 사인파
y   = y1 + y2 + y3 # + 1    # 임의의 파동 y (상수값을 추가하려면 주석을 해제할 것)

freqs = fftfreq(n,0.002)    # 필요한 모든 진동수를 만든다.

mask = freqs > 0    # 절반의 값을 무시
nwaves = freqs*n    # 도메인 길이에 따른 파수


fft_vals = fft(y)    # FFT 계산

fft_norm = fft_vals*(1.0/n)    # FFT 계산된 결과를 정규화


fft_theo = 2.0*abs(fft_norm)    # 푸리에 계수 계산


# 계산하고싶은 파수의 범위를 지정 (0~50 사이의 숫자를 입력)
wavenumber = int(input("input wavenumber (~50) : ",))

x0  = ones(n)
origin = fft_norm.real[0]*x0    # 상수부분인 푸리에 계수를 a0 더함

for k in range(1, wavenumber+1):    # 푸리에계수 an, bn을 이용해 IFFT 구현
    origin +=   2 * fft_norm.real[k] * cos(k*omg*x) + \
              (-2)* fft_norm.imag[k] * sin(k*omg*x)


# 1. 임의의 파동 y
plt.figure()
plt.subplot(211)
plt.plot(x, y, color='k', label='Original')
plt.plot(x, origin, color='g', label='IFFT')
plt.title("Original & IFFT Signal")
plt.legend()

# 2. 푸리에 계수
plt.subplot(212)
plt.bar(freqs[mask]*n, fft_theo[mask], label="true fft values")
plt.title("True FFT values")
plt.axhline(y=0, linewidth=1, color='k')
plt.legend()
plt.show()




# =============================================================================
# 
# ===========================================================================






%matplotlib auto

import numpy as np
import matplotlib.pyplot as plt
from numpy import arange, ones, pi
from scipy import cos, sin
from scipy.fftpack import fft, fftfreq, ifft

r_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/analysis_sigs/'
sig = np.load(r_path+'first_5.npy')
sig = sig.reshape(-1)

# sig = RM.values.reshape(-1)

x= np.arange(len(sig))
n = len(sig)
freqs = fftfreq(n,1)    # 필요한 모든 진동수를 만든다.

mask = freqs > 0    # 절반의 값을 무시
nwaves = freqs*n    # 도메인 길이에 따른 파수


fft_vals = fft(sig)    # FFT 계산

fft_norm = fft_vals*(1.0/n)    # FFT 계산된 결과를 정규화


fft_theo = 2.0*abs(fft_norm)    # 푸리에 계수 계산


# 계산하고싶은 파수의 범위를 지정 (0~50 사이의 숫자를 입력)
wavenumber = 100  #int(input("input wavenumber (~50) : ",))

x0  = ones(n)
origin = fft_norm.real[0]*x0    # 상수부분인 푸리에 계수를 a0 더함

for k in range(1, wavenumber+1):    # 푸리에계수 an, bn을 이용해 IFFT 구현
    origin +=   2 * fft_norm.real[k] * cos(k*omg*x) + \
              (-2)* fft_norm.imag[k] * sin(k*omg*x)


# 1.   
plt.figure(figsize=(16,7))
plt.subplot(211)
plt.plot(np.arange(len(sig)), sig, color='k', label='Original')
plt.plot(x, origin, color='g', label='IFFT')
plt.title("Original & IFFT Signal")
plt.legend()

# 2. 푸리에 계수
plt.subplot(212)
plt.bar(freqs[mask]*n, fft_theo[mask], label="true fft values",color='k')
plt.title("True FFT values")
# plt.axhspan(-0, .025, facecolor='0.5', alpha=0.2)
plt.axhline(y=0, linewidth=1, color='r')
plt.legend()
plt.show()








