# -*- coding: utf-8 -*-
#""""
#Created on Tue Feb 11 17:09:59 2020
#
#@author: user
#""""


import pandas as pd
import os

'''
test2 = open('D:/temp_task/20190703_01_h.cnv','r')
test2.close


with open('D:/temp_task/20190703_01_h.cnv', 'r') as the_file:
    test2_5 = [line.strip() for line in the_file.readlines()]
    height_line = test2_5[3]
    data = test2_5[8:]
'''



Directory = 'D:/temp_task/'

dir_list = os.listdir(Directory)
cnv_list = [file for file in dir_list if file.endswith('.cnv')]


CTD = pd.read_csv(Directory + cnv_list[0],
                    delim_whitespace=True,header=None)

# naming columns
CTD.columns = ['db1','Depth1','Temperature','Salinity','Density','Oxygen','Fluorescence','Scan_Count','flag']

# renaming columns         
CTD.rename(columns = {'db1': 'db', 'Depth1': 'Depth'}, inplace=True)

# changing values in the specific column
CTD['Depth'] = - CTD['Depth']

# changing index
# CTD = test3.set_index('db')
# CTD.reindex(index=test3['db'])
CTD.index = CTD['Depth']

# slicing variable
temp = CTD.iloc[:,2]
temp = CTD.loc[:,'Temperature']
db = CTD.loc[:,'db']
Depth = CTD.loc[:,'Depth']
Salinity = CTD.loc[:,'Salinity']

import GSW 


# ploting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# =============================================================================
#  %matplotlib inline
# =============================================================================

# 폰트확인


set(sorted([f.name for f in fm.fontManager.ttflist]))
[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# fm.get_fontconfig_fonts()
# font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
font_location = 'C:/Windows/Fonts/NanumBarunGothic.ttf' # For Windows
fontprop1 = fm.FontProperties(fname=font_location, size=20)

'''
plt.rcParams["figure.figsize"] = (10,4)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.labelsize'] = 12.
plt.rcParams['ytick.labelsize'] = 12.
'''

# make color
import numpy as np
from matplotlib import cm
from colorspacious import cspace_converter
from collections import OrderedDict


cmaps = OrderedDict()

# subplot 을 이용해 크기 조절

plt.figure(figsize=(10,10))
plt.title("t-s 다이어그램",fontproperties=fontprop1)
plt.scatter(temp,Salinity,c=-Depth,cmap=plt.cm.get_cmap('jet'))
plt.show()
        
        

'''
matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None,
 cmap=None, norm=None, vmin=None, vmax=None, alpha=None, 
 linewidths=None, verts=None, edgecolors=None, *,
 plotnonfinite=False, data=None, **kwargs)[source]
'''
        
        
        
        