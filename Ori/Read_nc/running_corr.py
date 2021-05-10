#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:15:04 2021

@author: caelus
"""

import pandas as pd
import numpy as np


KEYs = Sig_set_set.columns
print(KEYs)


Sig_set1 = Sig_set.first_5_2Y_Rm
Sig_set2 = Sig_set.EKE_qiu_2Y_Rm

np.corrcoef(Sig_set1[12:-12],Sig_set2[12:-12])


n,Corr_Matrix = 3,[]
factor = int(n*12/2)
for i in range(int(factor),len(Sig_set1)-factor):
    print(i)
    Corr_Matrix.append(np.corrcoef(Sig_set1[i-factor:i+factor],Sig_set2[i-factor:i+factor])[0,1])
    print(Corr_Matrix[i-factor],Sig_set.date[i])
    
    



w_path = '/home/caelus/dock_1/Working_hub/DATA_dep/Kuroshio/ALL/fig2/'



plt.figure(1,figsize=(16,7),dpi=80)
ax = plt.gca()
# plt.title('Yan Sun imf05',position=(0.5, 1.0+0.1),
#           fontweight='bold',fontsize='48')

# plt.plot(date,Sig_set_set.stats_model.values,label='statsmodels',color='darkgreen',linewidth=2.5)
# Draw Plot
plt.plot(Sig_set.date,Sig_set.first_5_2Y_Rm, label='SSH Diff (2Y Runing mean)',color='k',linewidth=2.5,zorder=10)
plt.plot(Sig_set.date,Sig_set.EKE_qiu_2Y_Rm, label='EKE qiu (2Y Runing mean)',color='darkred',linewidth=2.5,zorder=5)
plt.plot(Sig_set.date[factor:-factor],Corr_Matrix, label='Time-dependent Corr ('+str(n)+'Y)',color=[.77,.77,.77],
          zorder=0,linewidth=2.5)

# plt.plot(Sig_set.date[factor:-factor],Corr_Matrix, label=Sig_set1.name+' & '+Sig_set2.name,color=[.8,.8,.8],
#           zorder=0,linewidth=2.5)

plt.axhline(y=0,linewidth=2,color='k',linestyle='--')
plt.axhspan(-.5, .5, facecolor='0.5', alpha=0.2)
# Decoration
# plt.ylim(50,750)
xtick_location = Sig_set.index.tolist()[::12*2]
xtick_labels = Sig_set.date.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=30, fontsize=18, alpha=.7)
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=18, alpha=.7)
# Lighten borders
ax.tick_params(labelcolor='k', width=3)

plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower left',fontsize=18)
plt.grid(axis='y', alpha=.3)
if savefig:
    plt.savefig(w_path+'first_EKE_qiu_TD_Corr',dpi=150)
plt.show()







plt.figure(1,figsize=(16,10),dpi=80)
ax = plt.gca()
plt.plot(Sig_set.date[factor:-factor],Corr_Matrix, label=Sig_set1.name+' & '+Sig_set2.name,color=[.7,.7,.7],linewidth=2.5)
# plt.scatter(Sig_set.date[factor:-factor],Corr_Matrix)
plt.axhline(y=0,linewidth=3,color='k',linestyle='-.')
plt.axhspan(-.5, .5, facecolor='0.5', alpha=0.2)
xtick_location = Sig_set.index.tolist()[::12*2]
xtick_labels = Sig_set.date.tolist()[::12*2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=45, fontsize=22, alpha=.7)
plt.title("Time-dependent Corr (3Y window) ", fontsize=22)
plt.yticks(fontsize=22, alpha=.7)
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='upper right')
plt.grid(axis='y', alpha=.3)
# plt.savefig(w_path+'EKE_qiu_normalized',dpi=150)
plt.show()

















