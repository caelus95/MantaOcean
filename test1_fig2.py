#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:26:04 2021

@author: caelus
"""


# Compare sigs
plt.figure(1,figsize=(16,10),dpi=80)
ax = plt.gca()

plt.plot(Annual_mean.index,Annual_mean.ADT_index, label='SSH Diff (annual mean)',color='darkred',linewidth=2.5,zorder=5)
plt.scatter(Annual_mean.index,Annual_mean.ADT_index,color='darkred',zorder=6)

plt.plot(Annual_mean.index,Annual_mean.EKE_qiu_10_30_120_250, label='eddy (annual mean)',color='darkblue',linewidth=2.5,zorder=3)
plt.scatter(Annual_mean.index,Annual_mean.EKE_qiu_10_30_120_250,color='darkblue',zorder=4)

plt.plot(Annual_mean.index,Annual_mean.PDO, label='PDO (annual mean)',color=[.65,.65,.65],linewidth=2.5,zorder=1)
plt.scatter(Annual_mean.index,Annual_mean.PDO,color=[.65,.65,.65],zorder=2)

plt.bar(Annual_mean.index,Annual_mean.MEIv2, label='MEIv2',color=[.9,.9,.9],linewidth=2.5,zorder=0,alpha=.7)

# plt.plot(Annual_mean.index,Annual_mean.WP, label='WP (2Y Runing mean)',color='g',linewidth=2.5,zorder=3)
xtick_location = Annual_mean.index.tolist()[::2]
xtick_labels = Annual_mean.index.tolist()[::2]
plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=30, fontsize=16, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=20, alpha=.7,fontweight='bold')
# plt.title("Peak and Troughs of Air Passengers Traffic (1949 - 1969)", fontsize=22)
plt.xticks(fontsize=20, alpha=.7,fontweight='bold')
# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)
plt.legend(loc='lower right',fontsize=14)
plt.grid(axis='x', alpha=.3)
if savefig:
    plt.savefig(w_path1+'Index_annual',dpi=150)
plt.show()
