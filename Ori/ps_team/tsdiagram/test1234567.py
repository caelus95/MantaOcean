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

ii =0
A = open(Directory + cnv_list[ii])
b = A.readlines()
n=0
for i in b:
    if i[:5] == '*END*':
        header_num = n
        print('header_line = ',n)
        break
    else:
        header_num = 0
    n+=1
A.close()

CTD = pd.read_csv(Directory + cnv_list[ii],
                    delim_whitespace=True,header=header_num+1)

# naming subplot 
data_date = cnv_list[ii].split('_')[0]
data_station = cnv_list[ii].split('_')[1][:2]

def numdate2strdate(data_date):
    M = int(data_date[4:6])
    if M == 1:
        month = 'January'
    elif M == 2:
        month = 'February'
    elif M == 3:
        month = 'March'
    elif M == 4:
        month = 'April'
    elif M == 5:
        month = 'May'
    elif M == 6:
        month = 'June'
    elif M == 7:
        month = 'July'
    elif M == 8:
        month = 'August'
    elif M == 9:
        month = 'September'
    elif M == 10:
        month = 'October'
    elif M == 11:
        month = 'November'
    elif M == 12:
        month = 'December'
    else :
        print('DateName error/nPlease check data file name')    
    return data_date[:4] + '-' + month + '-' + data_date[6:8] 
        

subtitle = numdate2strdate(data_date) + ' St_' + data_station 

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

#import GSW 


# ploting
#import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# =============================================================================
#  %matplotlib inline
# =============================================================================

# 폰트확인

'''
set(sorted([f.name for f in fm.fontManager.ttflist]))
[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# fm.get_fontconfig_fonts()
# font_location = '/usr/share/fonts/truetype/nanum/NanumGothicOTF.ttf'
font_location = 'C:/Windows/Fonts/NanumBarunGothic.ttf' # For Windows
fontprop1 = fm.FontProperties(fname=font_location, size=32,fontweight = 'bold')
'''

# plt.rcParams.keys()
plt.rcParams["figure.figsize"] = (13,10)
plt.rcParams['lines.linewidth'] = 4
# plt.rcParams['lines.color'] = 'r'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '-.'
plt.rcParams["font.size"] = 28
plt.rcParams['xtick.labelsize'] = 14.
plt.rcParams['xtick.labeltop'] = True
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.major.width'] = 0
plt.rcParams['ytick.labelsize'] = 14.
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['ytick.major.width'] = 0
# subplot 을 이용해 크기 조절

# plt.figure(figsize=(10,10))
ax = plt.gca()
plt.title("T - S Diagram",position=(0.5, 1.0+0.12),fontweight='bold',fontsize='42')
plt.suptitle('('+subtitle+')',fontstyle='italic',
             position=(0.5, .96), fontsize=18)
f= plt.scatter(Salinity,temp,300,c=Depth,cmap=plt.cm.get_cmap('jet'))
plt.xlabel('Salinity(psu)',Fontweight='bold',fontstyle='italic')
plt.ylabel('Temperature(℃)',Fontweight='bold',fontstyle='italic')
plt.tick_params(axis="y", labelcolor="k",labelsize=20)
plt.tick_params(axis="x", labelcolor="k",labelsize=18)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
cax.tick_params(labelcolor="k",labelsize=16)
plt.colorbar(label='Depth(m)', cax=cax)
plt.show()


##############
        
plt.title("T - S Depth",position=(0.5, 1.0+0.17),fontweight='bold',fontsize='42')
plt.suptitle('('+subtitle+')',fontstyle='italic',
             position=(0.5, 1), fontsize=18)
plt.xlabel("Temperature(℃)",color = 'r',Fontweight='bold',fontstyle='italic')
plt.ylabel("Depth(m)", color="k",Fontweight='bold',fontstyle='italic')
plt.tick_params(axis="y", labelcolor="k",labelsize=20)
plt.tick_params(axis="x", labelcolor="r",labelsize=20)
plt.plot(temp,Depth,color='r')
plt.twiny()
plt.xlabel("Salinity(psu)", color="b",Fontweight='bold',fontstyle='italic',labelpad=10)
plt.tick_params(axis="x", labelcolor="b",labelsize=20)
plt.plot(Salinity,Depth,color='b',linewidth=4)


'''
fig = plt.figure(constrained_layout=True,figsize=(10,10))  
ax = fig.add_subplot(111)
#fig, ax = plt.subplots(constrained_layout=True,figsize=(10,10))
plt.title("T - S Depth\n(September 16 - October 30, 2012)",position=(0.5, 1.0+0.1),fontweight='bold')
ax.plot(Salinity,Depth,color='b')
ax.plot(temp,Depth,color='r')
plt.xlabel('Temporature')
#ax.set_xlabel('Salinity',Fontweight='bold',fontstyle='italic')
secax = ax.secondary_xaxis('bottom')
secax.set_xlabel('Salinity',Fontweight='bold',fontstyle='italic')
plt.show()
'''


        
        
        
        