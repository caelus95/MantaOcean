# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:54:14 2020

@author: psi36
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 

Directory = 'D:/temp_task/t_s_diagram/'

temp_lim = [10,32]
sal_lim = [30,36]
depth_lim = [-130,0] # t-s diagram

def dens_line(temp,sal):
    '''
    등밀도 선을 그리기 위한 작업
    return 값 : dens -> 2차원 자료 
    return 타입 : float64
    '''
    import gsw
    
    '''
    smin = sal.min() - (0.01 * sal.min())
    smax = sal.max() + (0.01 * sal.max())
    tmin = temp.min() - (0.1 * temp.max())
    tmax = temp.max() + (0.1 * temp.max())
    '''

    smin = sal_lim[0] - (0.01 * sal_lim[0])
    smax = sal_lim[1] + (0.01 * sal_lim[1])
    tmin = temp_lim[0] - (0.1 * temp_lim[0])
    tmax = temp_lim[1] + (0.1 * temp_lim[1])

    
    xdim = int(round((smax-smin)/0.1+1,0))
    ydim = int(round((tmax-tmin)+1,0))
    
    dens = np.zeros((ydim,xdim))
    
    ti = np.linspace(1,ydim-1,ydim)+tmin
    si = np.linspace(1,xdim-1,xdim)*0.1+smin
    
    for j in range(0,int(ydim)):
        for i in range(0, int(xdim)):
            dens[j,i]=gsw.rho(si[i],ti[j],0) - 1000
    
    return dens, si, ti
 
    
def numdate2strdate(data_date):
    '''
    변수 형식 : '년(4자리)월(2자리)일(2자리)_스테이션.cnv
    ex) 20190306_01.snv
    return 값 : 시간을 나타내는 문자열
    ex) 20190306_01.snv --> '2019-January-01 
    return 타입 : str
    '''
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
        print('DateName error\nPlease check data file name')    
    return data_date[:4] + '-' + month + '-' + data_date[6:8] 
        

dir_list = os.listdir(Directory)
cnv_list = [file for file in dir_list if file.endswith('.cnv')]
dens,si,ti = dens_line(temp_lim,sal_lim)

ii = 0
while ii <= len(cnv_list) -1 :
    A = open(Directory+cnv_list[ii])
    B = A.readlines()
    n=0
    for i in B:
        if i[:5] == '*END*' :
            header_num = n
            break
        else:
            header_num = 0
        n+=1

    A.close()
    
    data = pd.read_csv(Directory + cnv_list[ii],
                    delim_whitespace=True,header=header_num+1)
    
    data_date = cnv_list[ii].split('_')[0]
    data_station = cnv_list[ii].split('_')[1][:2]
    
    subtitle = numdate2strdate(data_date) + ' St_' + data_station 
    
    data.columns = ['db','Depth','Temperature','Salinity','Density','Oxygen','Fluorescence','Scan_Count','flag']
    
    data['Depth'] = - data['Depth']
    temp = data.loc[:,'Temperature']
    db = data.loc[:,'db']
    Depth = data.loc[:,'Depth']
    sal = data.loc[:,'Salinity']
    
    

    

    # default value (matplotlib)

    # plt.rcParams.keys()
    # plt.rcParams['lines.color'] = 'r'
    plt.rcParams["figure.figsize"] = (10,13)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['axes.grid'] = False
    plt.rcParams['grid.linestyle'] = '-.'
    plt.rcParams["font.size"] = 28
    plt.rcParams['xtick.labelsize'] = 14.
    plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.major.width'] = 0
    plt.rcParams['ytick.labelsize'] = 14.
    plt.rcParams['axes.linewidth'] = 3
    plt.rcParams['ytick.major.width'] = 0
   # plt.rcParams["savefig.jpeg_quality"] = 95 
    plt.rcParams["savefig.facecolor"] = 'white'
    
   
    
    # plot 1 --> t-s diagram
    plt.figure(1)
    ax = plt.gca()
    ax.grid(False)
    plt.title("T - S Diagram",position=(0.5, 1.0+0.09),fontweight='bold',fontsize='42')
    plt.suptitle('('+subtitle+')',fontstyle='italic',
             position=(0.5, .94), fontsize=18)
    plt.xlabel('Salinity(psu)',Fontweight='bold',fontstyle='italic',labelpad=20)
    plt.ylabel('Temperature(℃)',Fontweight='bold',fontstyle='italic')
    plt.tick_params(axis="y", labelcolor="k",labelsize=20)
    plt.tick_params(axis="x", labelcolor="k",labelsize=18)
    plt.ylim(temp_lim)
    plt.xlim(sal_lim)
    d = plt.contour(si,ti,dens, linestyles='dashdot',linewidths = 1 ,colors='grey')
    plt.clabel(d, fontsize=12, colors='b' ,inline=True, fmt='%1.0f') # Label every second level
   
    f= plt.scatter(sal,temp,300,c=Depth,cmap=plt.cm.get_cmap('jet'))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cax.set_ylabel('',{'fontsize':20,'fontweight':'bold','style':'italic'})
    cax.tick_params(labelcolor="k",labelsize=16)
    plt.colorbar(label='Depth(m)', cax=cax)
    plt.clim(depth_lim)
    plt.savefig('D:/temp_task/test/t_s_diagram_'+subtitle.replace('-','_').replace(' ','_')+'.jpg')
   # plt.show()
    plt.close()
      
    
    
    # plot 2 --> vertical temp & sal
    plt.figure(2)
    plt.title("T - S Depth",position=(0.5, 1.0+0.13),fontweight='bold',fontsize='42')
    plt.suptitle('('+subtitle+')',fontstyle='italic',
                 position=(0.5, .975), fontsize=18)
    plt.xlabel("Temperature(℃)",color = 'r',Fontweight='bold',fontstyle='italic')
    plt.ylabel("Depth(m)", color="k",Fontweight='bold',fontstyle='italic')
    plt.tick_params(axis="y", labelcolor="k",labelsize=20)
    plt.ylim(depth_lim) 
    plt.xlim(temp_lim) 
    plt.tick_params(axis="x", labelcolor="r",labelsize=20)
    plt.plot(temp,Depth,color='r')
    plt.twiny()
    plt.xlabel("Salinity(psu)", color="b",Fontweight='bold',fontstyle='italic',labelpad=10)
    plt.tick_params(axis="x", labelcolor="b",labelsize=20)
    plt.xlim(sal_lim) 
    plt.plot(sal,Depth,color='b',linewidth=4)
    plt.savefig('D:/temp_task/test/vertical_'+subtitle.replace('-','_').replace(' ','_')+'.jpg',bbox_inches='tight')
 #  plt.show()
    plt.close()
    
    ii+=1
    
    
    
    
    
    





