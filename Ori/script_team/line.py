# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:12:43 2019

@author: psi36
"""
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

del var
var = var_hyr

s,l,m = np.shape(var)
var_line1 = np.full([s,l],np.nan)
n=0
while n<= s-1:
    var_line1[n,:] = var[n,:,0]
    n+=1
    
line1 = np.transpose(var_line1)
    

hyr = [[1996, 1997, 1998, 2003, 2004, 2008, 2009, 2016],[3,4,5,10,11,15,16,23]]
lyr= [[1993, 2000, 2001, 2005, 2006, 2012, 2013, 2014],[0,7,8,12,13,19,20,21]]

strB = hyr[0]
n=0
for i in strB:
    strB[n] = str(i)+'Y'
    n+=1

'''
strA = list(range(17,36,2))
n=0
for i in strA:
    strA[n] = str(i)+'°N'
    n+=1
strB = list(range(1993,2017))
n=0
for i in strB:
    strB[n] = str(i)+'Y'
    n+=1

'''
# plot
plt.figure(figsize=(20, 10))
ax = plt.gca()
plt.xticks(np.arange(0, s+1, 1),strB,fontsize=18)
plt.yticks(np.arange(1, l+1, 2),strA,fontsize=18)
plt.xlabel('year', fontsize=24,style='italic')
plt.ylabel('latitude', fontsize=24,style='italic')
plt.contourf(line1,20,cmap=plt.cm.get_cmap('bwr'))
plt.title('HeatContent hyr (Longitude = 149.5°E)',{'fontsize':40,'fontweight':'bold'}, position=(0.5, 1.0+0.05) )
#plt.title(r'{\fontsize{30pt}{3em}\selectfont{}{Mean WRFv3.5 LHF\r}{\fontsize{18pt}{3em}\selectfont{}(September 16 - October 30, 2012)}')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2.5%", pad=0.1)
cax.tick_params(labelsize=15)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})

#label 

h = plt.colorbar(label='10^18_joules',cax=cax);